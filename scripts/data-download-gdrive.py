#!/usr/bin/env python3
"""
Download CelebA dataset from a custom Google Drive folder and process it.
This script downloads files from your friend's Google Drive folder instead of
the original source to avoid rate limits.
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import numpy as np
from torchvision.datasets import CelebA
from torchvision.transforms import PILToTensor
from tqdm import tqdm
import zipfile
import shutil
import subprocess

try:
    import gdown
except ImportError:
    print("ERROR: gdown is not installed. Install it with: pip install gdown")
    sys.exit(1)


def download_file_from_gdrive(file_id, output_path, quiet=False):
    """Download a single file from Google Drive using its file ID."""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(output_path), quiet=quiet)
        return True
    except Exception as e:
        print(f"Error downloading {output_path.name}: {e}")
        return False


def download_gdrive_folder(folder_url, output_dir, file_mapping=None):
    """
    Download files from a Google Drive folder.
    
    Args:
        folder_url: Google Drive folder URL or folder ID
        output_dir: Directory to save files
        file_mapping: Optional dict mapping file names to their Google Drive IDs
                     If None, will try to download entire folder
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract folder ID from URL if needed
    if "folders" in folder_url or "id=" in folder_url:
        if "id=" in folder_url:
            folder_id = folder_url.split("id=")[1].split("&")[0]
        else:
            folder_id = folder_url.split("folders/")[1].split("?")[0]
    else:
        # Assume it's just the folder ID
        folder_id = folder_url
    
    print(f"Downloading from Google Drive folder ID: {folder_id}")
    
    # If file mapping is provided, download specific files
    if file_mapping:
        print(f"Downloading {len(file_mapping)} files...")
        for filename, file_id in file_mapping.items():
            output_path = output_dir / filename
            if output_path.exists():
                print(f"  ✓ {filename} already exists, skipping...")
                continue
            
            print(f"  Downloading {filename}...")
            if download_file_from_gdrive(file_id, output_path, quiet=False):
                print(f"  ✓ Downloaded {filename}")
            else:
                print(f"  ✗ Failed to download {filename}")
                return False
    else:
        # Try to download entire folder
        print("Attempting to download entire folder...")
        try:
            gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", 
                                 output=str(output_dir), quiet=False, use_cookies=False)
        except Exception as e:
            print(f"Error downloading folder: {e}")
            print("\nPlease provide file IDs manually using --file-ids option")
            return False
    
    return True


def extract_zip_if_needed(zip_path, extract_to, use_system_unzip=True):
    """Extract zip file if it exists and hasn't been extracted. Supports resume."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    if not zip_path.exists():
        return False
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # Check if already fully extracted
    existing_files = set(f.name for f in extract_to.glob("*.jpg"))
    if len(existing_files) > 200000:  # CelebA has ~202k images
        print(f"  ✓ Images already extracted ({len(existing_files)} files found)")
        return True
    
    print(f"  Extracting {zip_path.name} to {extract_to}...")
    if len(existing_files) > 0:
        print(f"  Resuming extraction ({len(existing_files)} files already exist)...")
    
    # Try system unzip first (more memory efficient)
    if use_system_unzip:
        unzip_cmd = shutil.which("unzip")
        if unzip_cmd:
            try:
                print("  Using system unzip (more memory efficient)...")
                # Use unzip with -u (update) flag to skip existing files
                result = subprocess.run(
                    [unzip_cmd, "-u", "-q", str(zip_path), "-d", str(extract_to)],
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                if result.returncode == 0:
                    final_count = len(list(extract_to.glob("*.jpg")))
                    print(f"  ✓ Extraction complete ({final_count} files)")
                    return True
                else:
                    print(f"  System unzip failed, falling back to Python zipfile...")
                    print(f"  Error: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"  System unzip timed out, falling back to Python zipfile...")
            except Exception as e:
                print(f"  System unzip failed: {e}, falling back to Python zipfile...")
    
    # Fallback to Python zipfile (supports resume)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all members
            members = zip_ref.namelist()
            total = len(members)
            
            # Filter out already extracted files
            members_to_extract = [
                m for m in members 
                if Path(extract_to / m).name not in existing_files
            ]
            
            if len(members_to_extract) == 0:
                print(f"  ✓ All files already extracted")
                return True
            
            print(f"  Extracting {len(members_to_extract)}/{total} remaining files...")
            
            # Extract files one by one (memory efficient)
            extracted_count = 0
            for member in tqdm(members_to_extract, desc="Extracting", unit="file"):
                try:
                    # Extract single file
                    zip_ref.extract(member, extract_to)
                    extracted_count += 1
                    
                    # Flush to disk periodically
                    if extracted_count % 1000 == 0:
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    print(f"\n  Warning: Failed to extract {member}: {e}")
                    continue
            
            print(f"  ✓ Extraction complete ({extracted_count} files extracted)")
            return True
            
    except KeyboardInterrupt:
        print(f"\n  Extraction interrupted. Progress saved. Run again to resume.")
        return False
    except Exception as e:
        print(f"  ✗ Error extracting: {e}")
        print(f"  Partial extraction may be saved. Run again to resume.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download CelebA from Google Drive and process it"
    )
    
    parser.add_argument(
        "--gdrive-url",
        type=str,
        required=True,
        help="Google Drive folder URL or folder ID containing CelebA files"
    )
    
    parser.add_argument(
        "--file-ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional: Provide file IDs manually as 'filename:file_id' pairs. "
             "Example: 'img_align_celeba.zip:1ABC123' 'list_attr_celeba.txt:2XYZ456'"
    )
    
    parser.add_argument(
        "-j", "--nthreads",
        type=int,
        default=8,
        help="number of worker threads to use (default: 8)"
    )
    
    parser.add_argument(
        "-b", "--batchsize",
        type=int,
        default=128,
        help="batch_size for loading (default: 128, use 256+ for GPU)"
    )
    
    parser.add_argument(
        "-o", "--outdir",
        type=Path,
        default=Path("data"),
        help="output directory for processed data (default: data)"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="use GPU for processing if available"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing). If None, processes all."
    )
    
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, assume files already exist in celeba folder"
    )
    
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip zip extraction. Let torchvision extract on-the-fly (more memory efficient)"
    )
    
    args = parser.parse_args()
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        if args.batchsize < 256:
            args.batchsize = 256
            print(f"Increased batch size to {args.batchsize} for GPU processing")
    
    # Setup directories
    rootdir = args.outdir
    rootdir.mkdir(parents=True, exist_ok=True)
    
    dfolder = rootdir / "celeba"
    celeba_subfolder = dfolder / "celeba"
    celeba_subfolder.mkdir(parents=True, exist_ok=True)
    
    # Download files if not skipping
    if not args.skip_download:
        print(f"\n{'='*60}")
        print("Step 1: Downloading files from Google Drive")
        print(f"{'='*60}")
        
        # Parse file IDs if provided
        file_mapping = None
        if args.file_ids:
            file_mapping = {}
            for item in args.file_ids:
                if ":" in item:
                    filename, file_id = item.split(":", 1)
                    file_mapping[filename] = file_id
                else:
                    print(f"Warning: Invalid format '{item}', expected 'filename:file_id'")
        
        # Download files
        success = download_gdrive_folder(
            args.gdrive_url,
            celeba_subfolder,
            file_mapping=file_mapping
        )
        
        if not success:
            print("\nERROR: Failed to download some files.")
            print("You can:")
            print("  1. Check the Google Drive URL/folder ID")
            print("  2. Provide file IDs manually using --file-ids")
            print("  3. Download files manually and use --skip-download")
            return 1
        
        # Extract zip file if needed (unless skipping extraction)
        zip_file = celeba_subfolder / "img_align_celeba.zip"
        img_folder = celeba_subfolder / "img_align_celeba" / "img_align_celeba"
        
        if zip_file.exists():
            # Check if already extracted
            if img_folder.exists():
                existing_count = len(list(img_folder.glob("*.jpg")))
                if existing_count > 200000:
                    print(f"\n✓ Images already extracted ({existing_count} files)")
                elif existing_count > 0:
                    print(f"\n⚠ Partial extraction found ({existing_count} files)")
                    print(f"  Run this separately to complete extraction:")
                    print(f"  python scripts/extract_celeba.py {zip_file} -o {img_folder}")
                    if not args.skip_extraction:
                        print(f"\n  Attempting to continue extraction...")
                        extract_zip_if_needed(zip_file, img_folder, use_system_unzip=True)
                else:
                    if not args.skip_extraction:
                        print("\nExtracting zip file...")
                        print("  (This may take 10-20 minutes. If killed, run separately:)")
                        print(f"  python scripts/extract_celeba.py {zip_file} -o {img_folder}")
                        extract_zip_if_needed(zip_file, img_folder, use_system_unzip=True)
                    else:
                        print(f"\n⚠ Zip file exists but not extracted.")
                        print(f"  Extract manually: python scripts/extract_celeba.py {zip_file} -o {img_folder}")
        else:
            print(f"\n⚠ Zip file not found at {zip_file}")
    else:
        print("Skipping download (--skip-download flag set)")
        print(f"Expected files in: {celeba_subfolder}")
    
    # Verify required files exist
    required_files = [
        "img_align_celeba.zip",
        "list_attr_celeba.txt",
        "identity_CelebA.txt",
        "list_bbox_celeba.txt",
        "list_landmarks_align_celeba.txt",
        "list_eval_partition.txt"
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = celeba_subfolder / filename
        if not filepath.exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"\nWARNING: Missing files: {missing_files}")
        print("The script may fail. Please ensure all files are downloaded.")
    
    # Process dataset
    print(f"\n{'='*60}")
    print("Step 2: Processing CelebA dataset")
    print(f"{'='*60}")
    
    print("Loading and processing training split...")
    try:
        train_data = CelebA(
            str(dfolder),
            download=False,  # Don't download, we already have files
            transform=PILToTensor(),
            split="train"
        )
    except Exception as e:
        print(f"ERROR loading training data: {e}")
        print(f"Check that files are in: {dfolder / 'celeba'}")
        return 1
    
    print(f"Training samples: {len(train_data)}")
    
    # Limit dataset size if specified
    if args.max_samples and len(train_data) > args.max_samples:
        print(f"Limiting to {args.max_samples} samples for faster processing...")
        indices = torch.randperm(len(train_data))[:args.max_samples]
        train_data = torch.utils.data.Subset(train_data, indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.nthreads,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.nthreads > 0),
    )
    
    train_batches = []
    for b, (x, y) in enumerate(tqdm(train_loader, desc="Processing train")):
        if device.type == "cuda":
            x = x.to(device, non_blocking=True)
        train_batches.append(x.cpu().numpy())
    
    print("Loading and processing validation split...")
    try:
        val_data = CelebA(
            str(dfolder),
            download=False,
            transform=PILToTensor(),
            split="valid"
        )
    except Exception as e:
        print(f"ERROR loading validation data: {e}")
        return 1
    
    print(f"Validation samples: {len(val_data)}")
    
    # Limit validation size if specified
    if args.max_samples and len(val_data) > args.max_samples // 10:
        max_val = args.max_samples // 10
        print(f"Limiting validation to {max_val} samples...")
        indices = torch.randperm(len(val_data))[:max_val]
        val_data = torch.utils.data.Subset(val_data, indices)
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.nthreads,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.nthreads > 0),
    )
    
    val_batches = []
    for b, (x, y) in enumerate(tqdm(val_loader, desc="Processing val")):
        if device.type == "cuda":
            x = x.to(device, non_blocking=True)
        val_batches.append(x.cpu().numpy())
    
    # Save processed data
    print(f"\n{'='*60}")
    print("Step 3: Saving processed data")
    print(f"{'='*60}")
    
    train_x = np.concatenate(train_batches)
    train_output = dfolder / "train_data.npz"
    np.savez_compressed(train_output, data=train_x)
    print(f"✓ Wrote {train_output}")
    print(f"  Shape: {train_x.shape}, dtype: {train_x.dtype}")
    
    val_x = np.concatenate(val_batches)
    val_output = dfolder / "eval_data.npz"
    np.savez_compressed(val_output, data=val_x)
    print(f"✓ Wrote {val_output}")
    print(f"  Shape: {val_x.shape}, dtype: {val_x.dtype}")
    
    print(f"\n{'='*60}")
    print("✓ Complete! Dataset ready for training.")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

