(venv) [mubashir.m@c2193 vae_search]$ python scripts/data-download-gdrive.py \
    --gdrive-url "..." \
    --skip-download \
    -o data \
    -j 6 \
    --use-gpu \
    -b 256 \
    --checkpoint-dir data/celeba/.checkpoints
Using device: cuda
/home/mubashir.m/ondemand/data/sys/dashboard/batch_connect/sys/desktop-native-courses/output/e6d95e33-bb4f-4913-a299-51074c8c526e/vae_search/venv/lib64/python3.9/site-packages/torch/cuda/__init__.py:283: UserWarning: 
    Found GPU0 Tesla P100-PCIE-12GB which is of cuda capability 6.0.
    Minimum and Maximum cuda capability supported by this version of PyTorch is
    (7.0) - (12.0)
    
  warnings.warn(
/home/mubashir.m/ondemand/data/sys/dashboard/batch_connect/sys/desktop-native-courses/output/e6d95e33-bb4f-4913-a299-51074c8c526e/vae_search/venv/lib64/python3.9/site-packages/torch/cuda/__init__.py:304: UserWarning: 
    Please install PyTorch with a following CUDA
    configurations:  12.6 following instructions at
    https://pytorch.org/get-started/locally/
    
  warnings.warn(matched_cuda_warn.format(matched_arches))
/home/mubashir.m/ondemand/data/sys/dashboard/batch_connect/sys/desktop-native-courses/output/e6d95e33-bb4f-4913-a299-51074c8c526e/vae_search/venv/lib64/python3.9/site-packages/torch/cuda/__init__.py:326: UserWarning: 
Tesla P100-PCIE-12GB with CUDA capability sm_60 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
If you want to use the Tesla P100-PCIE-12GB GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(
GPU: Tesla P100-PCIE-12GB
Skipping download (--skip-download flag set)
Expected files in: data/celeba/celeba

============================================================
Checking extraction status...
============================================================
✓ Images already extracted (202599 files)
  Location: data/celeba/celeba/img_align_celeba
✓ Using image folder: data/celeba/celeba/img_align_celeba (202599 images)
✓ Images verified at expected location: data/celeba/celeba/img_align_celeba (202599 images)

============================================================
Step 2: Processing CelebA dataset
============================================================
Loading and processing training split...
Validating images (this may take a few minutes)...
Checking images: 100%|████████████████████████████████████████████████████████████████████| 162770/162770 [06:22<00:00, 425.50it/s]
✓ Validation complete: 162767 valid images, 3 skipped
Training samples: 162767 valid images
  ⚠ Skipped 3 corrupted/missing images
Processing train: 100%|██████████████████████████████████████████████████████████████████████████| 636/636 [00:33<00:00, 19.05it/s]

Saving training batches checkpoint...
✓ Saved checkpoint: data/celeba/.checkpoints/train_batches.npz
Loading and processing validation split...
Validating images (this may take a few minutes)...
Checking images: 100%|██████████████████████████████████████████████████████████████████████| 19867/19867 [00:26<00:00, 737.41it/s]
✓ Validation complete: 19867 valid images, 0 skipped
Validation samples: 19867 valid images
Processing val: 100%|██████████████████████████████████████████████████████████████████████████████| 78/78 [00:04<00:00, 18.57it/s]

Saving validation batches checkpoint...
✓ Saved checkpoint: data/celeba/.checkpoints/val_batches.npz

============================================================
Step 3: Saving processed data
============================================================
Concatenating training batches...
✓ Concatenation complete (6.7 seconds)
  Training array shape: (162767, 3, 218, 178), dtype: uint8
  Memory size: 17.65 GB

Saving training data (this may take 10-30 minutes)...
  Writing to: data/celeba/train_data.npz
✓ Wrote data/celeba/train_data.npz
  File size: 12.86 GB (compressed)
  Write time: 831.8 seconds (0.93 GB/min)

Concatenating validation batches...
✓ Concatenation complete (0.8 seconds)
  Validation array shape: (19867, 3, 218, 178), dtype: uint8
  Memory size: 2.15 GB

Saving validation data...
  Writing to: data/celeba/eval_data.npz
✓ Wrote data/celeba/eval_data.npz
  File size: 1.57 GB (compressed)
  Write time: 101.5 seconds (0.93 GB/min)

============================================================
✓ Complete! Dataset ready for training.
============================================================

============================================================
⚠ Summary of skipped images:
============================================================
  Training: 3 images skipped
    Skipped files: 036185.jpg, 066254.jpg, 085506.jpg
  Validation: 0 images skipped

  Total: 3 images skipped
  Successfully processed: 182634 valid images