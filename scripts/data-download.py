import sys
from torchvision.datasets import MNIST, CelebA, CIFAR10
import argparse
from pathlib import Path
import torch
import numpy as np
from torchvision.transforms import PILToTensor
from tqdm import tqdm
import os


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

        # Download and process training data
        print(f"Downloading and processing training split...")
        train_data = dataset(
            dfolder, download=True, transform=PILToTensor(), **train_kwarg
        )
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
        print(f"\nDownloading and processing validation split...")
        val_data = dataset(dfolder, download=True, transform=PILToTensor(), **val_kwarg)
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
