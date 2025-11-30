#!/usr/bin/env python3
"""
Standalone script to extract CelebA zip file.
Can be run separately in a long-running job or screen session.
Supports resume if interrupted.
"""

import argparse
from pathlib import Path
import subprocess
import zipfile
from tqdm import tqdm
import sys


def extract_with_unzip(zip_path, extract_to):
    """Extract using system unzip (most memory efficient)."""
    unzip_cmd = "unzip"
    try:
        print(f"Extracting {zip_path.name} using system unzip...")
        print(f"Target directory: {extract_to}")
        
        # Use -u flag to update (skip existing files)
        result = subprocess.run(
            [unzip_cmd, "-u", "-q", str(zip_path), "-d", str(extract_to)],
            capture_output=False,  # Show progress
            text=True
        )
        
        if result.returncode == 0:
            final_count = len(list(extract_to.glob("*.jpg")))
            print(f"\n✓ Extraction complete! ({final_count} files)")
            return True
        else:
            print(f"\n✗ Unzip failed with return code {result.returncode}")
            return False
            
    except FileNotFoundError:
        print("System unzip not found, falling back to Python zipfile...")
        return None
    except Exception as e:
        print(f"System unzip failed: {e}")
        return None


def extract_with_python(zip_path, extract_to):
    """Extract using Python zipfile (supports resume)."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # Check existing files
    existing_files = set(f.name for f in extract_to.glob("*.jpg"))
    print(f"Found {len(existing_files)} existing files")
    
    if len(existing_files) > 200000:
        print("✓ Already fully extracted!")
        return True
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            total = len(members)
            
            # Filter out already extracted
            members_to_extract = [
                m for m in members 
                if Path(extract_to / m).name not in existing_files
            ]
            
            if len(members_to_extract) == 0:
                print("✓ All files already extracted!")
                return True
            
            print(f"Extracting {len(members_to_extract)}/{total} remaining files...")
            
            extracted = 0
            for member in tqdm(members_to_extract, desc="Extracting", unit="file"):
                try:
                    zip_ref.extract(member, extract_to)
                    extracted += 1
                    
                    # Periodic cleanup
                    if extracted % 5000 == 0:
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    print(f"\nWarning: Failed to extract {member}: {e}")
                    continue
            
            final_count = len(list(extract_to.glob("*.jpg")))
            print(f"\n✓ Extraction complete! ({final_count} files)")
            return True
            
    except KeyboardInterrupt:
        print("\n\n⚠ Extraction interrupted. Progress saved.")
        print("Run again to resume extraction.")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract CelebA zip file. Supports resume if interrupted."
    )
    
    parser.add_argument(
        "zip_path",
        type=Path,
        help="Path to img_align_celeba.zip file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: same directory as zip)"
    )
    
    parser.add_argument(
        "--force-python",
        action="store_true",
        help="Force use of Python zipfile instead of system unzip"
    )
    
    args = parser.parse_args()
    
    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        print(f"Error: {zip_path} not found!")
        return 1
    
    if args.output:
        extract_to = Path(args.output)
    else:
        extract_to = zip_path.parent / "img_align_celeba"
    
    print("="*60)
    print("CelebA Zip Extraction Tool")
    print("="*60)
    print(f"Zip file: {zip_path}")
    print(f"Size: {zip_path.stat().st_size / (1024**3):.2f} GB")
    print(f"Extract to: {extract_to}")
    print("="*60)
    
    # Try system unzip first (unless forced to use Python)
    if not args.force_python:
        result = extract_with_unzip(zip_path, extract_to)
        if result is True:
            return 0
        elif result is False:
            print("\nFalling back to Python zipfile...")
    
    # Use Python zipfile
    result = extract_with_python(zip_path, extract_to)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())

