#!/bin/bash
# Script to train VAEGAN on CelebA dataset with GPU-optimized settings

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # Go to project root

# Load CUDA module if available
if command -v module &> /dev/null; then
    module load cuda/12.3.0 || module load cuda/12.8.0 || true
fi

# Activate virtual environment
source venv/bin/activate

# Check if CelebA data exists
if [ ! -f "data/celeba/train_data.npz" ] || [ ! -f "data/celeba/eval_data.npz" ]; then
    echo "CelebA data not found. Please download it first:"
    echo "  python data-download-gdrive.py celeba -o data"
    exit 1
fi

echo "=========================================="
echo "Starting VAEGAN training on CelebA (GPU)"
echo "=========================================="
echo "Model config: configs/celeba/vaegan_config.json"
echo "Training config: configs/celeba/vaegan_training_config_gpu.json"
echo "=========================================="

# Verify GPU availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo "Warning: Could not verify GPU"

# Start training with wandb
python training.py \
    --dataset celeba \
    --model_name vaegan \
    --model_config configs/celeba/vaegan_config.json \
    --training_config configs/celeba/vaegan_training_config_gpu.json \
    --nn convnet \
    --use_wandb \
    --wandb_project vaegan-celeba \
    --wandb_entity benchmark_team

echo "=========================================="
echo "Training completed!"
echo "=========================================="


