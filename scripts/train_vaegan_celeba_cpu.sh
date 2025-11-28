#!/bin/bash
# Script to train VAEGAN on CelebA dataset with CPU-optimized settings

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source ../../venv/bin/activate

# Check if CelebA data exists
if [ ! -f "data/celeba/train_data.npz" ] || [ ! -f "data/celeba/eval_data.npz" ]; then
    echo "CelebA data not found. Please download it first:"
    echo "  python data-download.py celeba -o data"
    exit 1
fi

echo "=========================================="
echo "Starting VAEGAN training on CelebA (CPU)"
echo "=========================================="
echo "Model config: configs/celeba/vaegan_config.json"
echo "Training config: configs/celeba/vaegan_training_config_cpu.json"
echo "=========================================="

# Start training
python training.py \
    --dataset celeba \
    --model_name vaegan \
    --model_config configs/celeba/vaegan_config.json \
    --training_config configs/celeba/vaegan_training_config_cpu.json \
    --nn convnet

echo "=========================================="
echo "Training completed!"
echo "=========================================="


