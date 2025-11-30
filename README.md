# Variational Autoencoder Research Project: Comprehensive Analysis of VAE Architectures

**Authors:** Hamza Mubashir and Alexsei Krotov

## Overview

This repository presents a comprehensive research project focused on implementing, training, and evaluating multiple Variational Autoencoder (VAE) architectures on the CelebA dataset. We conducted an in-depth comparative analysis across four distinct VAE variants: **VAE**, **BetaVAE**, **VAEGAN**, and **VQVAE**, systematically exploring how architectural choices and hyperparameter variations impact model performance, reconstruction quality, and generation capabilities.

Our work represents a significant research effort involving:
- **Custom implementations** and careful hyperparameter tuning for each architecture
- **GPU compute infrastructure** setup on the Explorer cluster
- **Hugging Face integration** for model storage and sharing
- **Weights & Biases (wandb)** integration for comprehensive experiment tracking
- **Rigorous evaluation** using multiple perceptual and quantitative metrics
- **Reproducible experimental framework** with detailed documentation

## Project Structure

```
vae_search/
├── training.py                          # Main training script for all VAE models
├── evaluate_vqvae.py                    # VQVAE evaluation script with comprehensive metrics
├── evaluate_vaegan.py                   # VAEGAN evaluation script with comprehensive metrics
├── data-download-gdrive.py              # CelebA dataset downloader from Google Drive
├── celebA_betavae.ipynb                 # BetaVAE analysis notebook with detailed experiments
│
├── configs/                             # Model and training configurations
│   ├── celeba/                          # CelebA-specific configurations
│   │   ├── vqvae_config.json            # VQVAE model configuration
│   │   ├── vqvae_training_config_gpu.json    # GPU-optimized training config
│   │   ├── vqvae_training_config_cpu.json    # CPU-optimized training config
│   │   ├── vaegan_config.json           # VAEGAN model configuration
│   │   ├── vaegan_training_config_gpu.json   # GPU-optimized training config
│   │   ├── vaegan_training_config_cpu.json   # CPU-optimized training config
│   │   └── [other model configs]       # Configurations for VAE, BetaVAE, etc.
│   ├── mnist/                           # MNIST dataset configurations
│   ├── cifar10/                         # CIFAR-10 dataset configurations
│   └── [other datasets]/                # Additional dataset configurations
│
├── training_shell_scripts/              # Shell scripts for easy training execution
│   ├── train_vqvae_celeba_gpu.sh        # GPU training script for VQVAE
│   ├── train_vqvae_celeba_cpu.sh        # CPU training script for VQVAE
│   ├── train_vaegan_celeba_gpu.sh        # GPU training script for VAEGAN
│   └── train_vaegan_celeba_cpu.sh        # CPU training script for VAEGAN
│
├── pythae/                              # PyTorch VAE library (modified/forked)
│   ├── src/pythae/                      # Core library source code
│   └── scripts/                         # Utility scripts (distributed training, etc.)
│
├── VQVAE_METRICS_DOCUMENTATION.md      # Comprehensive VQVAE metrics documentation
├── VAEGAN_METRICS_DOCUMENTATION.md      # Comprehensive VAEGAN metrics documentation
├── PRETRAINED_MODELS_INFO.md           # Information about pretrained models
├── ENVIRONMENT_SETUP.md                 # Environment setup instructions
└── requirements.txt                    # Python package dependencies
```

## Key Files Description

### Python Scripts

#### `training.py`
The core training script that supports training all VAE variants (VAE, BetaVAE, VAEGAN, VQVAE, and many others) on multiple datasets (MNIST, CIFAR-10, CelebA). Features include:
- **Flexible model selection** via command-line arguments
- **Configurable architectures** (convnet, resnet) for different datasets
- **Wandb integration** for experiment tracking
- **Automatic model checkpointing** and best model saving
- **Support for adversarial training** (VAEGAN, AAE, FactorVAE)
- **Coupled optimizer support** for models requiring separate optimizers

**Usage:**
```bash
python training.py \
    --dataset celeba \
    --model_name vqvae \
    --model_config configs/celeba/vqvae_config.json \
    --training_config configs/celeba/vqvae_training_config_gpu.json \
    --nn convnet \
    --use_wandb \
    --wandb_project vqvae-celeba
```

#### `evaluate_vqvae.py`
Comprehensive evaluation script for VQVAE models that computes:
- **Reconstruction metrics**: MSE, BCE (for comparison), VQ Loss, Total Loss
- **Perceptual metrics**: LPIPS, SSIM, GMSD
- **Generation metrics**: Self-reconstruction quality of generated samples
- **Wandb logging** for all metrics

**Key Features:**
- Loads models from checkpoint directories
- Processes evaluation data in batches
- Generates samples from codebook for generation metrics
- Detailed documentation of why certain metrics (like KL divergence) don't apply to VQVAE

**Usage:**
```bash
python evaluate_vqvae.py \
    --checkpoint_path path/to/checkpoint/final_model \
    --data_root data \
    --dataset celeba \
    --n_eval_samples 10000 \
    --use_wandb \
    --wandb_project vqvae-evaluation
```

#### `evaluate_vaegan.py`
Comprehensive evaluation script for VAEGAN models that computes:
- **Reconstruction metrics**: MSE, BCE, Feature-space Recon Loss, KL divergence, ELBO
- **Adversarial metrics**: Encoder Loss, Decoder Loss, Discriminator Loss
- **Perceptual metrics**: LPIPS, SSIM, GMSD
- **Generation metrics**: Quality of samples from prior distribution

**Key Features:**
- Computes KL divergence directly from encoder outputs
- Tracks all three loss components (encoder, decoder, discriminator)
- Generates samples from standard normal prior
- Comprehensive documentation of VAEGAN's unique architecture

**Usage:**
```bash
python evaluate_vaegan.py \
    --checkpoint_path path/to/checkpoint/final_model \
    --data_root data \
    --dataset celeba \
    --n_eval_samples 10000 \
    --use_wandb \
    --wandb_project vaegan-evaluation
```

#### `data-download-gdrive.py`
Custom dataset downloader for CelebA that:
- Downloads from Google Drive (avoiding rate limits from official sources)
- Processes and preprocesses images
- Saves data in the required `.npz` format
- Handles dataset extraction and organization

**Usage:**
```bash
python data-download-gdrive.py celeba -o data
```

### Jupyter Notebook

#### `celebA_betavae.ipynb`
Comprehensive analysis notebook for BetaVAE experiments featuring:
- **Data loading and preprocessing** for CelebA
- **Model definition and training** with different beta values
- **Reconstruction metric computation** (MSE, BCE, KL, ELBO, LPIPS, SSIM, GMSD)
- **Generation quality evaluation** using self-reconstruction metrics
- **Visualization** of reconstructions and generated samples
- **Hyperparameter exploration** and analysis

This notebook demonstrates our careful experimental methodology and thorough analysis of BetaVAE performance across different configurations.

### Configuration Files (`configs/`)

The `configs/celeba/` folder contains carefully tuned configurations for each model:

- **Model configs** (`*_config.json`): Define model architecture, latent dimensions, loss functions, and model-specific hyperparameters
- **Training configs** (`*_training_config_*.json`): Define training parameters including:
  - Batch sizes (optimized for GPU vs CPU)
  - Learning rates (tuned for each architecture)
  - Number of epochs
  - Checkpoint saving frequency
  - Data loader workers
  - CUDA settings

**Key Configurations:**
- `vqvae_training_config_gpu.json`: Optimized for GPU with batch size 128, 50 epochs, wandb integration
- `vaegan_training_config_gpu.json`: GPU-optimized with coupled optimizers for encoder/decoder/discriminator
- CPU configs: Smaller batch sizes, fewer epochs, optimized for CPU training

### Training Shell Scripts (`training_shell_scripts/`)

Convenient shell scripts that handle:
- **Environment setup** (CUDA module loading, virtual environment activation)
- **Data validation** (checking for required dataset files)
- **GPU verification** (checking CUDA availability)
- **Automated training** with proper wandb integration
- **Error handling** and informative output

**Example:**
```bash
./training_shell_scripts/train_vqvae_celeba_gpu.sh
```

### Documentation Files

#### `VQVAE_METRICS_DOCUMENTATION.md`
Comprehensive documentation explaining:
- Why VQVAE doesn't use BCE (hardcoded to MSE)
- Why KL divergence doesn't apply (deterministic vs probabilistic)
- How VQ Loss replaces KL as regularization
- Architectural differences from BetaVAE
- Metric interpretation guide

#### `VAEGAN_METRICS_DOCUMENTATION.md`
Detailed documentation covering:
- Feature-space reconstruction loss (vs pixel-wise)
- Adversarial training components
- Loss component breakdown (encoder, decoder, discriminator)
- Comparison with BetaVAE and VQVAE
- Training dynamics and coupled optimizers

## Research Methodology

### Experimental Design

Our research approach involved:

1. **Systematic Model Implementation**: We implemented and trained four distinct VAE architectures:
   - **VAE**: Baseline probabilistic autoencoder
   - **BetaVAE**: Beta-weighted KL divergence for disentanglement
   - **VAEGAN**: Adversarial training with feature-space reconstruction
   - **VQVAE**: Vector quantization for discrete latent representations

2. **Careful Hyperparameter Tuning**: Each model received dedicated attention:
   - Learning rate optimization
   - Batch size tuning for GPU efficiency
   - Architecture-specific hyperparameters (beta values, codebook sizes, adversarial scales)
   - Training duration optimization

3. **Comprehensive Evaluation**: We developed custom evaluation scripts that compute:
   - **Quantitative metrics**: MSE, BCE, KL divergence, VQ Loss
   - **Perceptual metrics**: LPIPS, SSIM, GMSD
   - **Generation quality**: Self-reconstruction metrics
   - **Model-specific metrics**: Discriminator loss (VAEGAN), Codebook utilization (VQVAE)

### Infrastructure Setup

#### GPU Compute on Explorer Cluster
We set up and configured GPU compute infrastructure on the Explorer cluster, including:
- CUDA module loading and environment configuration
- Batch job submission scripts
- Resource allocation optimization
- Multi-GPU training support (where applicable)

#### Hugging Face Integration
All trained models are stored and shared via Hugging Face Hub:
- Model versioning and organization
- Easy model loading for evaluation
- Reproducible model sharing
- Integration with evaluation scripts

#### Weights & Biases (wandb) Integration
Comprehensive experiment tracking including:
- Training loss curves (reconstruction, regularization, total)
- Validation metrics over time
- Hyperparameter logging
- Model checkpoint tracking
- Comparative analysis across experiments

### Analysis and Insights

Our careful analysis revealed:
- **Architectural trade-offs**: How different regularization approaches (KL, VQ Loss, Adversarial) affect reconstruction quality
- **Perceptual vs pixel metrics**: Understanding when feature-space losses outperform pixel losses
- **Generation quality**: Systematic evaluation of how well each model generates realistic samples
- **Training dynamics**: Observation of how different architectures converge and stabilize

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Virtual environment manager (venv, conda, etc.)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd vae_search
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install pythae library first (required dependency)
cd pythae
pip install -e .
cd ..

# Install all other dependencies
pip install -r requirements.txt
```

### Step 4: Download Dataset

```bash
# Download CelebA dataset from Google Drive
python data-download-gdrive.py celeba -o data
```

This will create `data/celeba/` with `train_data.npz` and `eval_data.npz` files.

### Step 5: Configure Wandb (Optional but Recommended)

```bash
# Login to wandb
wandb login

# Your API key will be prompted
```

## Usage Examples

### Training VQVAE on CelebA (GPU)

```bash
# Using the shell script (recommended)
./training_shell_scripts/train_vqvae_celeba_gpu.sh

# Or directly with Python
python training.py \
    --dataset celeba \
    --model_name vqvae \
    --model_config configs/celeba/vqvae_config.json \
    --training_config configs/celeba/vqvae_training_config_gpu.json \
    --nn convnet \
    --use_wandb \
    --wandb_project vqvae-celeba \
    --wandb_entity your_entity
```

### Training VAEGAN on CelebA (GPU)

```bash
# Using the shell script
./training_shell_scripts/train_vaegan_celeba_gpu.sh

# Or directly with Python
python training.py \
    --dataset celeba \
    --model_name vaegan \
    --model_config configs/celeba/vaegan_config.json \
    --training_config configs/celeba/vaegan_training_config_gpu.json \
    --nn convnet \
    --use_wandb \
    --wandb_project vaegan-celeba
```

### Evaluating a Trained Model

```bash
# Evaluate VQVAE
python evaluate_vqvae.py \
    --checkpoint_path my_models_on_celeba/VQVAE_training_YYYY-MM-DD_HH-MM-SS/final_model \
    --data_root data \
    --dataset celeba \
    --n_eval_samples 10000 \
    --use_wandb \
    --wandb_project vqvae-evaluation

# Evaluate VAEGAN
python evaluate_vaegan.py \
    --checkpoint_path my_models_on_celeba/VAEGAN_training_YYYY-MM-DD_HH-MM-SS/final_model \
    --data_root data \
    --dataset celeba \
    --n_eval_samples 10000 \
    --use_wandb \
    --wandb_project vaegan-evaluation
```

### Running BetaVAE Analysis Notebook

```bash
# Start Jupyter
jupyter notebook

# Open celebA_betavae.ipynb
# Follow the cells to reproduce our BetaVAE experiments
```

## Reproducibility

All experiments are fully reproducible:

1. **Fixed random seeds** in training and evaluation scripts
2. **Version-controlled configurations** for all hyperparameters
3. **Documented data preprocessing** steps
4. **Model checkpoints** saved with full configuration
5. **Evaluation scripts** that can load any checkpoint and reproduce metrics

## Key Contributions

1. **Comprehensive Implementation**: Full implementations of VAE, BetaVAE, VAEGAN, and VQVAE with careful attention to architectural details
2. **Rigorous Evaluation Framework**: Custom evaluation scripts with multiple perceptual and quantitative metrics
3. **Infrastructure Setup**: GPU compute configuration, Hugging Face integration, and wandb tracking
4. **Detailed Documentation**: Extensive documentation explaining architectural differences and metric interpretations
5. **Reproducible Research**: Complete experimental framework that can be easily reproduced and extended

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{vae_search,
  author = {Mubashir, Hamza and Krotov, Alexsei},
  title = {Variational Autoencoder Research: Comprehensive Analysis of VAE Architectures},
  year = {2025},
  url = {<repository-url>}
}
```

## License

[Specify your license here]

## Acknowledgments

- PyTorch VAE (pythae) library for the base implementations
- Explorer cluster team for GPU compute resources
- Hugging Face for model hosting infrastructure
- Weights & Biases for experiment tracking platform

## Contact

For questions or collaborations, please contact:
- Hamza Mubashir: [email]
- Alexsei Krotov: [email]
