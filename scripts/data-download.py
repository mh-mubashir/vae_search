import sys
from torchvision.datasets import MNIST, CelebA, CIFAR10
import argparse
from pathlib import Path
import torch
import numpy as np
from torchvision.transforms import PILToTensor
from tqdm import tqdm
import os


def check_celeba_files_exist(dfolder):
    """Check if CelebA dataset files already exist to avoid re-downloading."""
    celeba_subfolder = dfolder / "celeba" / "celeba"
    required_files = [
        "img_align_celeba.zip",
        "list_attr_celeba.txt",
        "identity_CelebA.txt",
        "list_bbox_celeba.txt",
        "list_landmarks_align_celeba.txt",
        "list_eval_partition.txt"
    ]
    
    # Check if zip file exists (main indicator)
    zip_file = celeba_subfolder / "img_align_celeba.zip"
    if zip_file.exists() and zip_file.stat().st_size > 1000000000:  # > 1GB
        print(f"✓ Found existing CelebA files in {celeba_subfolder}")
        print(f"  Zip file size: {zip_file.stat().st_size / (1024**3):.2f} GB")
        # Check if extracted images exist
        img_folder = celeba_subfolder / "img_align_celeba"
        if img_folder.exists() and len(list(img_folder.glob("*.jpg"))) > 100000:
            print(f"  Found {len(list(img_folder.glob('*.jpg')))} extracted images")
            return True
        return True  # Zip exists, can extract if needed
    
    return False


def main():

    parser = argparse.ArgumentParser(
        description="python script to download datasets which are available with torchvision"
    )

    parser.add_argument(
        "-j", "--nthreads", type=int, default=None, 
        help="number of worker threads to use (default: auto-detect CPU count)"
    )
    parser.add_argument(
        "-b", "--batchsize", type=int, default=128, 
        help="batch_size for loading (default: 128, use 256+ for GPU)"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("."),
        help="the base folder in which to store the output",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="use GPU for processing if available",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="process and save data in chunks to reduce memory usage",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="force re-download even if files exist (use with caution for CelebA)",
    )
    parser.add_argument(
        "dataset",
        nargs="+",
        help="datasets to download (possible values: MNIST, CelebA, CIFAR10)",
    )
    args = parser.parse_args()
    if not "dataset" in args:
        print("dataset argument not found in", args)
        parser.print_help()
        return 1

    # Auto-detect number of workers if not specified
    if args.nthreads is None:
        args.nthreads = min(os.cpu_count() or 1, 8)  # Use up to 8 cores
        print(f"Auto-detected {args.nthreads} CPU cores, using {args.nthreads} workers")

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Increase batch size for GPU
        if args.batchsize < 256:
            args.batchsize = 256
            print(f"Increased batch size to {args.batchsize} for GPU processing")

    tv_datasets = {"mnist": MNIST, "celeba": CelebA, "cifar10": CIFAR10}
    rootdir = args.outdir
    if not rootdir.exists():
        print(f"creating root folder {rootdir}")
        rootdir.mkdir(parents=True)

    for dname in args.dataset:
        if dname.lower() not in tv_datasets.keys():
            print(f"{dname} not available for download yet. skipping.")
            continue

        dfolder = rootdir / dname
        dataset = tv_datasets[dname]
        if "celeba" in dname.lower():
            train_kwarg = {"split": "train"}
            val_kwarg = {"split": "val"}
        else:
            train_kwarg = {"train": True}
            val_kwarg = {"train": False}

        print(f"\n{'='*60}")
        print(f"Processing {dname.upper()} dataset")
        print(f"{'='*60}")

        # Check if files already exist (especially for CelebA to avoid Google Drive limits)
        should_download = True
        if "celeba" in dname.lower() and not args.force_download:
            if check_celeba_files_exist(dfolder):
                print("Skipping download - files already exist (to avoid Google Drive rate limits)")
                print("  Use --force-download to re-download if needed")
                should_download = False
            else:
                print("Files not found, will attempt download...")
        elif args.force_download:
            print("Force download enabled - will attempt to re-download files")

        # Download and process training data
        print(f"Loading and processing training split...")
        try:
            # First try without download if files might exist
            if not should_download:
                try:
                    train_data = dataset(
                        dfolder, download=False, transform=PILToTensor(), **train_kwarg
                    )
                    print("✓ Successfully loaded existing files")
                except (FileNotFoundError, RuntimeError) as e:
                    if "not found" in str(e).lower() or "not downloaded" in str(e).lower():
                        print("Files not complete, attempting download...")
                        train_data = dataset(
                            dfolder, download=True, transform=PILToTensor(), **train_kwarg
                        )
                    else:
                        raise
            else:
                train_data = dataset(
                    dfolder, download=True, transform=PILToTensor(), **train_kwarg
                )
        except Exception as e:
            if "Too many users" in str(e) or "FileURLRetrievalError" in str(e):
                print("\n" + "="*60)
                print("ERROR: Google Drive rate limit hit!")
                print("="*60)
                print("The CelebA files appear to be downloaded but verification failed.")
                print(f"\nExpected location: {dfolder / 'celeba' / 'celeba'}")
                print("\nSolutions:")
                print("  1. Wait 24 hours and try again")
                print("  2. Check if files exist manually:")
                print(f"     ls -lh {dfolder / 'celeba' / 'celeba'}")
                print("  3. If files exist, try running with --skip-verification")
                print("  4. Use alternative download method")
                raise
            else:
                raise
        print(f"Training samples: {len(train_data)}")
        
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=args.batchsize, 
            shuffle=False, 
            num_workers=args.nthreads,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.nthreads > 0),
        )

        # Process training data in chunks to save memory
        train_chunks = []
        train_batch_list = []
        for b, (x, y) in enumerate(tqdm(train_loader, desc="Processing train")):
            if device.type == "cuda":
                x = x.to(device, non_blocking=True)
            train_batch_list.append(x.cpu().numpy())
            
            # Process in chunks to avoid memory issues
            if len(train_batch_list) * args.batchsize >= args.chunk_size:
                chunk = np.concatenate(train_batch_list)
                train_chunks.append(chunk)
                train_batch_list = []
        
        # Add remaining batches
        if train_batch_list:
            chunk = np.concatenate(train_batch_list)
            train_chunks.append(chunk)
        
        train_x = np.concatenate(train_chunks) if len(train_chunks) > 1 else train_chunks[0]
        print(f"Saving training data: shape {train_x.shape}, dtype {train_x.dtype}")
        np.savez_compressed(dfolder / "train_data.npz", data=train_x)
        print(f"✓ Wrote {dfolder / 'train_data.npz'}")

        # Download and process validation data
        print(f"\nLoading and processing validation split...")
        try:
            # Use same download setting as training
            if not should_download:
                try:
                    val_data = dataset(dfolder, download=False, transform=PILToTensor(), **val_kwarg)
                except (FileNotFoundError, RuntimeError) as e:
                    if "not found" in str(e).lower() or "not downloaded" in str(e).lower():
                        val_data = dataset(dfolder, download=True, transform=PILToTensor(), **val_kwarg)
                    else:
                        raise
            else:
                val_data = dataset(dfolder, download=True, transform=PILToTensor(), **val_kwarg)
        except Exception as e:
            if "Too many users" in str(e) or "FileURLRetrievalError" in str(e):
                print("\n" + "="*60)
                print("ERROR: Google Drive rate limit hit!")
                print("="*60)
                print(f"Expected location: {dfolder / 'celeba' / 'celeba'}")
                raise
            else:
                raise
        print(f"Validation samples: {len(val_data)}")
        
        val_loader = torch.utils.data.DataLoader(
            val_data, 
            batch_size=args.batchsize, 
            shuffle=False, 
            num_workers=args.nthreads,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.nthreads > 0),
        )

        # Process validation data in chunks
        val_chunks = []
        val_batch_list = []
        for b, (x, y) in enumerate(tqdm(val_loader, desc="Processing val")):
            if device.type == "cuda":
                x = x.to(device, non_blocking=True)
            val_batch_list.append(x.cpu().numpy())
            
            if len(val_batch_list) * args.batchsize >= args.chunk_size:
                chunk = np.concatenate(val_batch_list)
                val_chunks.append(chunk)
                val_batch_list = []
        
        if val_batch_list:
            chunk = np.concatenate(val_batch_list)
            val_chunks.append(chunk)
        
        val_x = np.concatenate(val_chunks) if len(val_chunks) > 1 else val_chunks[0]
        print(f"Saving validation data: shape {val_x.shape}, dtype {val_x.dtype}")
        np.savez_compressed(dfolder / "eval_data.npz", data=val_x)
        print(f"✓ Wrote {dfolder / 'eval_data.npz'}")

    return 0


if __name__ == "__main__":
    rv = main()
    sys.exit(rv)
