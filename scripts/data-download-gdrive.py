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
from torch.utils.data import Dataset
from PIL import Image
import PIL
import warnings

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


class RobustCelebA(Dataset):
    """
    Wrapper around CelebA dataset that skips corrupted or missing images.
    Tracks and reports how many images were skipped.
    """
    def __init__(self, celeba_dataset, skip_errors=True, validate_upfront=True):
        self.celeba_dataset = celeba_dataset
        self.skip_errors = skip_errors
        self.skipped_indices = set()
        self.skipped_count = 0
        self.skipped_files = []  # Track which files were skipped
        
        if validate_upfront:
            # Pre-validate all images and build valid index mapping
            print("Validating images (this may take a few minutes)...")
            self.valid_indices = []
            self.index_mapping = {}  # Maps new index to original CelebA index
            
            img_base_path = os.path.join(
                celeba_dataset.root,
                celeba_dataset.base_folder,
                "img_align_celeba"
            )
            
            for idx in tqdm(range(len(celeba_dataset)), desc="Checking images"):
                try:
                    img_filename = celeba_dataset.filename[idx]
                    img_path = os.path.join(img_base_path, img_filename)
                    
                    # Check if file exists
                    if not os.path.exists(img_path):
                        self.skipped_indices.add(idx)
                        self.skipped_count += 1
                        self.skipped_files.append(img_filename)
                        continue
                    
                    # Try to open and verify image (quick check)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # Verify it's a valid image
                        
                        # If we get here, image is valid
                        self.index_mapping[len(self.valid_indices)] = idx
                        self.valid_indices.append(idx)
                    except Exception as e:
                        # Image is corrupted
                        self.skipped_indices.add(idx)
                        self.skipped_count += 1
                        self.skipped_files.append(img_filename)
                        continue
                        
                except Exception as e:
                    # Some other error
                    self.skipped_indices.add(idx)
                    self.skipped_count += 1
                    if idx < len(celeba_dataset.filename):
                        self.skipped_files.append(celeba_dataset.filename[idx])
                    continue
            
            print(f"✓ Validation complete: {len(self.valid_indices)} valid images, {self.skipped_count} skipped")
        else:
            # Lazy validation - validate as we go
            self.valid_indices = list(range(len(celeba_dataset)))
            self.index_mapping = {i: i for i in range(len(celeba_dataset))}
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Map our index to the original CelebA index
        original_idx = self.index_mapping[idx]
        
        # Get the item from original dataset
        # Wrap in try-except as a safety net
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.celeba_dataset[original_idx]
            except (PIL.UnidentifiedImageError, FileNotFoundError, OSError) as e:
                # Image is corrupted or missing
                if self.skip_errors:
                    # Track this as skipped
                    if original_idx not in self.skipped_indices:
                        self.skipped_indices.add(original_idx)
                        self.skipped_count += 1
                        if original_idx < len(self.celeba_dataset.filename):
                            self.skipped_files.append(self.celeba_dataset.filename[original_idx])
                    
                    # Try next valid image
                    if idx + 1 < len(self.valid_indices):
                        return self.__getitem__(idx + 1)
                    elif len(self.valid_indices) > 0 and idx > 0:
                        return self.__getitem__(idx - 1)
                    else:
                        raise RuntimeError(f"No valid images found. Skipped {self.skipped_count} images.")
                else:
                    raise
            except Exception as e:
                # Other errors - retry or raise
                if attempt < max_retries - 1:
                    continue
                else:
                    if self.skip_errors:
                        warnings.warn(f"Error loading image at index {original_idx}: {e}")
                        # Try to return a valid image
                        if len(self.valid_indices) > 0:
                            # Try a different index
                            alt_idx = (idx + 1) % len(self.valid_indices)
                            if alt_idx != idx:
                                return self.__getitem__(alt_idx)
                    raise


def extract_zip_if_needed(zip_path, extract_to, use_system_unzip=True):
    """Extract zip file if it exists and hasn't been extracted. Supports resume."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    if not zip_path.exists():
        return False
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # Check if already fully extracted - check both direct and nested locations
    # The zip may create extract_to/img_align_celeba/ structure
    direct_files = set(f.name for f in extract_to.glob("*.jpg"))
    nested_folder = extract_to / "img_align_celeba"
    nested_files = set()
    if nested_folder.exists():
        nested_files = set(f.name for f in nested_folder.glob("*.jpg"))
    
    total_existing = len(direct_files) + len(nested_files)
    if total_existing > 200000:  # CelebA has ~202k images
        if len(nested_files) > len(direct_files):
            print(f"  ✓ Images already extracted ({len(nested_files)} files found in nested location)")
        else:
            print(f"  ✓ Images already extracted ({len(direct_files)} files found)")
        return True
    
    print(f"  Extracting {zip_path.name} to {extract_to}...")
    if total_existing > 0:
        print(f"  Resuming extraction ({total_existing} files already exist)...")
    
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
                    # Check both direct and nested locations
                    direct_count = len(list(extract_to.glob("*.jpg")))
                    nested_count = 0
                    if nested_folder.exists():
                        nested_count = len(list(nested_folder.glob("*.jpg")))
                    final_count = direct_count + nested_count
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
            
            # Filter out already extracted files (check both locations)
            all_existing = direct_files | nested_files
            members_to_extract = [
                m for m in members 
                if Path(extract_to / m).name not in all_existing
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
    else:
        print("Skipping download (--skip-download flag set)")
        print(f"Expected files in: {celeba_subfolder}")
    
    # Always check extraction status (regardless of download skip)
    print(f"\n{'='*60}")
    print("Checking extraction status...")
    print(f"{'='*60}")
    
    zip_file = celeba_subfolder / "img_align_celeba.zip"
    # The zip contains img_align_celeba folder, so extracted path is nested
    img_folder = celeba_subfolder / "img_align_celeba" / "img_align_celeba"
    # Also check the direct extraction location (in case zip structure differs)
    img_folder_alt = celeba_subfolder / "img_align_celeba"
    
    if zip_file.exists():
        # Check both possible locations
        existing_count = 0
        existing_count_alt = 0
        
        if img_folder.exists():
            existing_count = len(list(img_folder.glob("*.jpg")))
        if img_folder_alt.exists():
            existing_count_alt = len(list(img_folder_alt.glob("*.jpg")))
        
        # Use the location with more images (likely the correct one)
        if existing_count > existing_count_alt and existing_count > 200000:
            print(f"✓ Images already extracted ({existing_count} files)")
            print(f"  Location: {img_folder}")
        elif existing_count_alt > existing_count and existing_count_alt > 200000:
            print(f"✓ Images already extracted ({existing_count_alt} files)")
            print(f"  Location: {img_folder_alt}")
            img_folder = img_folder_alt  # Update to use this path
        elif existing_count > 0 or existing_count_alt > 0:
            total_found = max(existing_count, existing_count_alt)
            print(f"⚠ Partial extraction found ({total_found} files)")
            if existing_count > existing_count_alt:
                print(f"  Location: {img_folder}")
            else:
                print(f"  Location: {img_folder_alt}")
                img_folder = img_folder_alt
            print(f"  Run this separately to complete extraction:")
            print(f"  python scripts/extract_celeba.py {zip_file} -o {celeba_subfolder / 'img_align_celeba'}")
            if not args.skip_extraction:
                print(f"\n  Attempting to continue extraction...")
                extract_zip_if_needed(zip_file, celeba_subfolder / "img_align_celeba", use_system_unzip=True)
        else:
            if not args.skip_extraction:
                print("\nExtracting zip file...")
                print("  (This may take 10-20 minutes. If killed, run separately:)")
                print(f"  python scripts/extract_celeba.py {zip_file} -o {celeba_subfolder / 'img_align_celeba'}")
                extract_zip_if_needed(zip_file, celeba_subfolder / "img_align_celeba", use_system_unzip=True)
            else:
                print(f"⚠ Zip file exists but not extracted.")
                print(f"  Extract manually: python scripts/extract_celeba.py {zip_file} -o {celeba_subfolder / 'img_align_celeba'}")
                print(f"\n  Or use system unzip:")
                print(f"  cd {celeba_subfolder} && unzip -u img_align_celeba.zip -d img_align_celeba")
    else:
        print(f"⚠ Zip file not found at {zip_file}")
        print(f"  Make sure the zip file is downloaded first.")
    
    # Verify extraction before proceeding - check both locations
    final_count = 0
    if img_folder.exists():
        final_count = len(list(img_folder.glob("*.jpg")))
    elif img_folder_alt.exists():
        final_count = len(list(img_folder_alt.glob("*.jpg")))
        img_folder = img_folder_alt  # Use alternative location
    
    if final_count < 200000:
        print(f"\n{'='*60}")
        print("ERROR: Images not fully extracted!")
        print(f"{'='*60}")
        print(f"Found only {final_count} images (need ~202,599)")
        print(f"\nPlease extract the zip file first:")
        print(f"  python scripts/extract_celeba.py {zip_file} -o {celeba_subfolder / 'img_align_celeba'}")
        print(f"\nOr use system unzip:")
        print(f"  cd {celeba_subfolder} && unzip -u img_align_celeba.zip -d img_align_celeba")
        return 1
    
    if not img_folder.exists():
        print(f"\n{'='*60}")
        print("ERROR: Images folder not found!")
        print(f"{'='*60}")
        print(f"Checked: {img_folder}")
        print(f"Also checked: {img_folder_alt}")
        print(f"\nPlease extract the zip file first:")
        print(f"  python scripts/extract_celeba.py {zip_file} -o {celeba_subfolder / 'img_align_celeba'}")
        return 1
    
    print(f"✓ Using image folder: {img_folder} ({final_count} images)")
    
    # torchvision CelebA expects images at {root}/celeba/img_align_celeba/
    # where root is the folder passed to CelebA(). Since we pass dfolder (data/celeba),
    # it will look for data/celeba/celeba/img_align_celeba/
    # If the zip creates data/celeba/celeba/img_align_celeba/img_align_celeba/,
    # we need to move files to the expected location
    expected_img_folder = celeba_subfolder / "img_align_celeba"
    
    # Check if images are in nested location (img_align_celeba/img_align_celeba/)
    nested_img_folder = celeba_subfolder / "img_align_celeba" / "img_align_celeba"
    
    if nested_img_folder.exists():
        nested_count = len(list(nested_img_folder.glob("*.jpg")))
        if nested_count > 200000:
            print(f"\n⚠ Images found in nested location: {nested_img_folder}")
            print(f"  torchvision expects: {expected_img_folder}")
            
            # Check if expected location already has images
            expected_count = 0
            if expected_img_folder.exists():
                # Check if it's the nested folder itself or has images
                if expected_img_folder.is_dir():
                    expected_count = len(list(expected_img_folder.glob("*.jpg")))
            
            if expected_count < 200000:
                print(f"  Moving files from nested location to expected location...")
                try:
                    # If expected folder doesn't exist or is empty, move nested folder contents
                    if not expected_img_folder.exists() or expected_count == 0:
                        # Move all files from nested to expected location
                        if expected_img_folder.exists() and expected_count == 0:
                            # Remove empty expected folder
                            try:
                                expected_img_folder.rmdir()
                            except:
                                pass
                        
                        # Move contents from nested folder to expected location
                        # The nested folder is inside the expected folder, so move contents up one level
                        print(f"  Moving {nested_count} image files from nested to expected location...")
                        expected_img_folder.mkdir(parents=True, exist_ok=True)
                        
                        moved_count = 0
                        for img_file in tqdm(nested_img_folder.glob("*.jpg"), desc="Moving files", unit="file"):
                            try:
                                target_path = expected_img_folder / img_file.name
                                if not target_path.exists():
                                    shutil.move(str(img_file), str(target_path))
                                    moved_count += 1
                                else:
                                    # File already exists, skip
                                    img_file.unlink()
                                    moved_count += 1
                            except Exception as e:
                                print(f"\n  Warning: Failed to move {img_file.name}: {e}")
                                continue
                        
                        # Remove empty nested folder
                        try:
                            nested_img_folder.rmdir()
                        except:
                            pass
                        
                        print(f"  ✓ Moved {moved_count} files to expected location")
                        
                        # Update img_folder to expected location
                        img_folder = expected_img_folder
                        final_count = len(list(img_folder.glob("*.jpg")))
                        print(f"  ✓ Verification: {final_count} images in expected location")
                    else:
                        print(f"  Expected location already has {expected_count} images, keeping as is")
                except Exception as e:
                    print(f"  ✗ Error moving files: {e}")
                    print(f"  You may need to manually move files from:")
                    print(f"    {nested_img_folder}")
                    print(f"  to:")
                    print(f"    {expected_img_folder}")
                    return 1
    
    # Final verification - ensure expected location has images
    if expected_img_folder.exists():
        expected_count = len(list(expected_img_folder.glob("*.jpg")))
        if expected_count > 200000:
            img_folder = expected_img_folder
            final_count = expected_count
            print(f"✓ Images verified at expected location: {img_folder} ({final_count} images)")
        elif nested_img_folder.exists():
            nested_count = len(list(nested_img_folder.glob("*.jpg")))
            if nested_count > 200000:
                print(f"\n⚠ WARNING: Images still in nested location!")
                print(f"  Nested: {nested_img_folder} ({nested_count} images)")
                print(f"  Expected: {expected_img_folder} ({expected_count} images)")
                print(f"  torchvision will fail. Please move files manually or re-extract.")
                return 1
    
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
        celeba_train = CelebA(
            str(dfolder),
            download=False,  # Don't download, we already have files
            transform=PILToTensor(),
            split="train"
        )
        # Wrap with robust dataset that skips corrupted images
        train_data = RobustCelebA(celeba_train, skip_errors=True)
    except Exception as e:
        print(f"ERROR loading training data: {e}")
        print(f"Check that files are in: {dfolder / 'celeba'}")
        return 1
    
    print(f"Training samples: {len(train_data)} valid images")
    if train_data.skipped_count > 0:
        print(f"  ⚠ Skipped {train_data.skipped_count} corrupted/missing images")
    
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
        celeba_val = CelebA(
            str(dfolder),
            download=False,
            transform=PILToTensor(),
            split="valid"
        )
        # Wrap with robust dataset that skips corrupted images
        val_data = RobustCelebA(celeba_val, skip_errors=True)
    except Exception as e:
        print(f"ERROR loading validation data: {e}")
        return 1
    
    print(f"Validation samples: {len(val_data)} valid images")
    if val_data.skipped_count > 0:
        print(f"  ⚠ Skipped {val_data.skipped_count} corrupted/missing images")
    
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
    
    # Report skipped images summary
    total_skipped = train_data.skipped_count + val_data.skipped_count
    if total_skipped > 0:
        print(f"\n{'='*60}")
        print("⚠ Summary of skipped images:")
        print(f"{'='*60}")
        print(f"  Training: {train_data.skipped_count} images skipped")
        if train_data.skipped_count > 0 and len(train_data.skipped_files) <= 10:
            print(f"    Skipped files: {', '.join(train_data.skipped_files[:10])}")
        elif train_data.skipped_count > 10:
            print(f"    First 10 skipped files: {', '.join(train_data.skipped_files[:10])}")
        
        print(f"  Validation: {val_data.skipped_count} images skipped")
        if val_data.skipped_count > 0 and len(val_data.skipped_files) <= 10:
            print(f"    Skipped files: {', '.join(val_data.skipped_files[:10])}")
        elif val_data.skipped_count > 10:
            print(f"    First 10 skipped files: {', '.join(val_data.skipped_files[:10])}")
        
        print(f"\n  Total: {total_skipped} images skipped")
        print(f"  Successfully processed: {len(train_data) + len(val_data)} valid images")
        print(f"{'='*60}")
    else:
        print(f"\n✓ All images processed successfully!")
        print(f"  Training: {len(train_data)} images")
        print(f"  Validation: {len(val_data)} images")
    
    return 0


if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

