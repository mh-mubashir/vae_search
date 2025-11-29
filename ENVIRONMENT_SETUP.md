# FiDIP Environment Setup

This guide explains how to set up the FiDIP environment on the Northeastern HPC cluster.

## Prerequisites

- Access to Northeastern HPC cluster
- CUDA-compatible GPU node access

## Module Loading

Before working with FiDIP, you need to load the required modules:

```bash
# Load CUDA 12.3.0 (compatible with CUDA 12.2)
module load cuda/12.3.0

# Load miniconda3 for environment management
module load miniconda3/24.11.1
```


## GPU Access

### Request GPU Node
To access GPU resources, request a compute node with GPU:

```bash
# Request any available GPU (24 hours)
srun --gres=gpu:1 --time=24:00:00 --pty bash

# Request specific GPU type (e.g., V100) for 24 hours
srun --gres=gpu:v100:1 --time=24:00:00 --pty bash

# Request A100 GPU for 24 hours
srun --gres=gpu:a100:1 --time=24:00:00 --pty bash
```

### Verify GPU Access
Once on a GPU node, verify GPU availability:

```bash
# Check GPU status
nvidia-smi

# Test PyTorch GPU access (after environment setup)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Environment Activation

```bash
# Activate the environment
conda activate venv

# Or use conda run
conda run -n venv python your_script.py
```

## Available GPU Types

The cluster has various GPU types available:
- **A100**: High-performance GPUs (nodes d1026, d1028, d1029, etc.)
- **V100**: Tesla V100 GPUs (nodes c2204-c2207, d1002, d1007, etc.)
- **H100**: Latest generation GPUs (nodes d4041)
- **H200**: Newest generation GPUs (nodes d4052-d4055)
- **L40/L40s**: Professional GPUs (nodes d3230, d3231, d4042-d4044, etc.)
- **A5000/A6000**: Professional GPUs (nodes d3165, d3166, d3168, etc.)
- **T4**: Entry-level GPUs (node d1025)
- **P100**: Older generation GPUs (nodes c2184-c2195)

## Troubleshooting

### GPU Access Issues
- Ensure you're on a GPU compute node (not login node)
- Check GPU availability with `nvidia-smi`
- Verify CUDA modules are loaded
- Test PyTorch GPU detection

### Memory Issues
If installation fails due to memory limits:
- Use `--no-cache-dir` flag with pip
- Install packages in smaller batches
- Request a compute node with more memory

## Quick Start Commands

```bash
# 1. Load modules
module load cuda/12.3.0
module load miniconda3/24.11.1

# 2. Request GPU node (24 hours)
srun --gres=gpu:1 --time=24:00:00 --pty bash

# 3. Create environment
conda env create -f venv

# 4. Activate and test
conda activate venv
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Notes

- Always load modules before creating/using environments
- GPU access requires requesting compute nodes via SLURM
- Login nodes don't have GPU access
- Environment creation may take time due to large package downloads