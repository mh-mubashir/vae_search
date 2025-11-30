# Comprehensive Documentation

**Authors:** Hamza Mubashir and Alexsei Krotov

This document provides comprehensive documentation for the VAE research project, covering environment setup, dataset preparation, model evaluation metrics, and pretrained model information.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Pretrained Models](#pretrained-models)
4. [Evaluation Metrics Overview](#evaluation-metrics-overview)
5. [VQVAE Metrics](#vqvae-metrics)
6. [VAEGAN Metrics](#vaegan-metrics)
7. [Model Architecture Comparison](#model-architecture-comparison)
8. [References](#references)

---

## Environment Setup

### Prerequisites

- Access to Northeastern HPC cluster (Explorer cluster)
- CUDA-compatible GPU node access
- Python 3.8 or higher

### Module Loading

Before working with the project, load the required modules on the HPC cluster:

```bash
# Load CUDA 12.3.0 (compatible with CUDA 12.2)
module load cuda/12.3.0

# Load miniconda3 for environment management
module load miniconda3/24.11.1
```

### GPU Access

#### Request GPU Node

To access GPU resources, request a compute node with GPU:

```bash
# Request any available GPU (24 hours)
srun --gres=gpu:1 --time=24:00:00 --pty bash

# Request specific GPU type (e.g., V100) for 24 hours
srun --gres=gpu:v100:1 --time=24:00:00 --pty bash

# Request A100 GPU for 24 hours
srun --gres=gpu:a100:1 --time=24:00:00 --pty bash
```

#### Verify GPU Access

Once on a GPU node, verify GPU availability:

```bash
# Check GPU status
nvidia-smi

# Test PyTorch GPU access (after environment setup)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Available GPU Types

The Explorer cluster has various GPU types available:
- **A100**: High-performance GPUs (nodes d1026, d1028, d1029, etc.)
- **V100**: Tesla V100 GPUs (nodes c2204-c2207, d1002, d1007, etc.)
- **H100**: Latest generation GPUs (nodes d4041)
- **H200**: Newest generation GPUs (nodes d4052-d4055)
- **L40/L40s**: Professional GPUs (nodes d3230, d3231, d4042-d4044, etc.)
- **A5000/A6000**: Professional GPUs (nodes d3165, d3166, d3168, etc.)
- **T4**: Entry-level GPUs (node d1025)
- **P100**: Older generation GPUs (nodes c2184-c2195)

### Environment Activation

```bash
# Activate the environment
conda activate venv

# Or use conda run
conda run -n venv python your_script.py
```

### Troubleshooting

#### GPU Access Issues
- Ensure you're on a GPU compute node (not login node)
- Check GPU availability with `nvidia-smi`
- Verify CUDA modules are loaded
- Test PyTorch GPU detection

#### Memory Issues
If installation fails due to memory limits:
- Use `--no-cache-dir` flag with pip
- Install packages in smaller batches
- Request a compute node with more memory

### Quick Start Commands

```bash
# 1. Load modules
module load cuda/12.3.0
module load miniconda3/24.11.1

# 2. Request GPU node (24 hours)
srun --gres=gpu:1 --time=24:00:00 --pty bash

# 3. Create environment (if not already created)
conda env create -f venv

# 4. Activate and test
conda activate venv
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Important Notes:**
- Always load modules before creating/using environments
- GPU access requires requesting compute nodes via SLURM
- Login nodes don't have GPU access
- Environment creation may take time due to large package downloads

---

## Dataset Preparation

### Downloading CelebA Dataset

The project uses a custom Google Drive downloader to avoid rate limits from official sources.

#### Basic Usage

```bash
python data-download-gdrive.py celeba -o data
```

#### Advanced Options

```bash
python data-download-gdrive.py \
    --gdrive-url "your_google_drive_url" \
    --skip-download \
    -o data \
    -j 6 \
    --use-gpu \
    -b 256 \
    --checkpoint-dir data/celeba/.checkpoints
```

**Parameters:**
- `-o data`: Output directory for processed data
- `-j 6`: Number of parallel workers for processing
- `--use-gpu`: Use GPU acceleration for image processing
- `-b 256`: Batch size for processing
- `--checkpoint-dir`: Directory for saving processing checkpoints
- `--skip-download`: Skip download step if files already exist

### Dataset Processing

The script performs the following steps:

1. **Download**: Downloads CelebA dataset from Google Drive
2. **Extraction**: Extracts images from archive
3. **Validation**: Validates image integrity (checks for corrupted files)
4. **Processing**: Processes images into batches
5. **Saving**: Saves processed data as `.npz` files

### Expected Output

After successful processing, you should have:

```
data/
└── celeba/
    ├── train_data.npz      # Training data (~12-17 GB)
    ├── eval_data.npz       # Evaluation data (~1.5-2 GB)
    └── .checkpoints/        # Processing checkpoints (optional)
```

**Data Format:**
- Shape: `(n_images, 3, height, width)` in uint8 format [0-255]
- Training: ~162,770 images
- Evaluation: ~19,867 images
- Images are preprocessed and ready for training

### Processing Notes

- The script automatically handles corrupted/missing images
- Processing time: ~10-30 minutes depending on hardware
- GPU acceleration significantly speeds up image processing
- Checkpoints allow resuming interrupted processing

---

## Pretrained Models

### Accessing Pretrained Models

Pretrained models can be loaded using the pythae library:

```python
from pythae.models import AutoModel

# Load from Hugging Face Hub
my_model = AutoModel.load_from_hf_hub(hf_hub_path="clementchadebec/model_name")

# Load from local path
my_model = AutoModel.load_from_folder("path/to/checkpoint/final_model")
```

### Available Pretrained Models

The following models are available on Hugging Face Hub (from the pythae reproducibility section):

| Model | Dataset | Hugging Face Link |
|-------|---------|-------------------|
| VAE | Binary MNIST | [clementchadebec/reproduced_vae](https://huggingface.co/clementchadebec/reproduced_vae) |
| VAMP (K=500) | Binary MNIST | [clementchadebec/reproduced_vamp](https://huggingface.co/clementchadebec/reproduced_vamp) |
| SVAE | Dyn. Binarized MNIST | [clementchadebec/reproduced_svae](https://huggingface.co/clementchadebec/reproduced_svae) |
| PoincareVAE (Wrapped) | MNIST | [clementchadebec/reproduced_wrapped_poincare_vae](https://huggingface.co/clementchadebec/reproduced_wrapped_poincare_vae) |
| IWAE (n_samples=50) | Binary MNIST | [clementchadebec/reproduced_iwae](https://huggingface.co/clementchadebec/reproduced_iwae) |
| MIWAE (M=8, K=8) | Dyn. Binarized MNIST | [clementchadebec/reproduced_miwae](https://huggingface.co/clementchadebec/reproduced_miwae) |
| PIWAE (M=8, K=8) | Dyn. Binarized MNIST | [clementchadebec/reproduced_piwae](https://huggingface.co/clementchadebec/reproduced_piwae) |
| CIWAE (beta=0.05) | Dyn. Binarized MNIST | [clementchadebec/reproduced_ciwae](https://huggingface.co/clementchadebec/reproduced_ciwae) |
| HVAE (n_lf=4) | Binary MNIST | [clementchadebec/reproduced_hvae](https://huggingface.co/clementchadebec/reproduced_hvae) |
| BetaTCVAE | DSPRITES | [clementchadebec/reproduced_beta_tc_vae](https://huggingface.co/clementchadebec/reproduced_beta_tc_vae) |
| RAE_L2 | MNIST | [clementchadebec/reproduced_rae_l2](https://huggingface.co/clementchadebec/reproduced_rae_l2) |
| RAE_GP | MNIST | [clementchadebec/reproduced_rae_gp](https://huggingface.co/clementchadebec/reproduced_rae_gp) |
| WAE | CELEBA 64 | [clementchadebec/reproduced_wae](https://huggingface.co/clementchadebec/reproduced_wae) |
| AAE | CELEBA 64 | [clementchadebec/reproduced_aae](https://huggingface.co/clementchadebec/reproduced_aae) |

### Models NOT Available as Pretrained

The following models **do NOT** have pretrained versions available on Hugging Face Hub:

- ❌ **VQ-VAE** (Vector Quantized VAE)
- ❌ **VAEGAN** (Variational Autoencoder GAN)
- ❌ **BetaVAE**
- ❌ **FactorVAE**
- ❌ **DisentangledBetaVAE**
- ❌ **VAE_LinNF**
- ❌ **VAE_IAF**
- ❌ **MSSSIM_VAE**
- ❌ **INFOVAE_MMD**
- ❌ **RHVAE**
- ❌ **HRQVAE**
- ❌ **HVAE** (on CelebA)

### Important Notes

1. **No CelebA pretrained models for VQ-VAE or VAEGAN**: Neither VQ-VAE nor VAEGAN have pretrained models available on HuggingFace Hub, especially not for CelebA dataset.

2. **Training required**: For VQ-VAE and VAEGAN on CelebA, you must train the models yourself using the provided training scripts.

3. **How to check for models**: 
   - Check the reproducibility README: `pythae/examples/scripts/reproducibility/README.md`
   - Search HuggingFace Hub directly: `https://huggingface.co/models?search=clementchadebec`
   - Try loading: `AutoModel.load_from_hf_hub("clementchadebec/model_name")` - will raise error if not found

4. **Available CelebA models**: Only **WAE** and **AAE** have pretrained models available for CelebA dataset.

---

## Evaluation Metrics Overview

Our evaluation framework computes comprehensive metrics to assess model performance across multiple dimensions:

### Quantitative Metrics

- **MSE** (Mean Squared Error): Pixel-wise reconstruction quality
- **BCE** (Binary Cross-Entropy): Alternative reconstruction metric (for comparison)
- **KL Divergence**: Latent space regularization (probabilistic models only)
- **VQ Loss**: Vector quantization quality (VQVAE only)
- **ELBO** (Evidence Lower Bound): Combined reconstruction and regularization

### Perceptual Metrics

- **LPIPS** (Learned Perceptual Image Patch Similarity): Perceptual similarity using VGG network
- **SSIM** (Structural Similarity Index): Structural similarity between images
- **GMSD** (Gradient Magnitude Similarity Deviation): Gradient-based similarity

### Model-Specific Metrics

- **VAEGAN**: Encoder Loss, Decoder Loss, Discriminator Loss
- **VQVAE**: VQ Loss (commitment + quantization)

### Generation Metrics

All models are evaluated on generation quality by:
1. Generating samples from the model's prior/codebook
2. Reconstructing those generated samples
3. Computing metrics between original and reconstructed generations

This measures model consistency and generation quality.

---

## VQVAE Metrics

### Overview

VQVAE (Vector Quantized VAE) is a **deterministic model** that uses vector quantization instead of probabilistic sampling. This fundamental difference affects which metrics are applicable.

### Metrics Computed

1. **MSE** (Mean Squared Error)
   - Primary reconstruction loss used by VQVAE
   - Measures pixel-wise squared difference between original and reconstructed images
   - Lower is better (0 = perfect reconstruction)

2. **BCE** (Binary Cross-Entropy)
   - Computed for comparison purposes only
   - **Note**: VQVAE does NOT use BCE in its loss function (it's hardcoded to MSE)
   - Included to allow comparison with BetaVAE metrics

3. **VQ_Loss** (Vector Quantization Loss)
   - Replaces KL divergence in VQVAE
   - Measures how well the codebook represents encoder outputs
   - Consists of commitment loss + quantization loss
   - Lower indicates better codebook utilization

4. **Total_Loss**
   - Sum of reconstruction loss (MSE) and VQ loss
   - The overall training objective

5. **LPIPS, SSIM, GMSD**
   - Perceptual metrics (same as other models)

### Why BCE and KL Are Different in VQVAE

#### Binary Cross-Entropy (BCE)

**In BetaVAE:**
- BetaVAE can use either MSE or BCE as reconstruction loss
- Controlled by `reconstruction_loss` parameter in config

**In VQVAE:**
- **VQVAE only uses MSE** - it's hardcoded in the loss function
- No option to use BCE reconstruction loss
- The evaluation script computes BCE for comparison purposes only

**Code Reference:**
```python
# VQVAE loss function (from pythae/src/pythae/models/vq_vae/vq_vae_model.py)
def loss_function(self, recon_x, x, quantizer_output):
    recon_loss = F.mse_loss(  # Always MSE, no BCE option
        recon_x.reshape(x.shape[0], -1), 
        x.reshape(x.shape[0], -1), 
        reduction="none"
    ).sum(dim=-1)
    vq_loss = quantizer_output.loss
    return (recon_loss + vq_loss).mean(dim=0), recon_loss.mean(dim=0), vq_loss.mean(dim=0)
```

#### Kullback-Leibler (KL) Divergence

**Why KL Doesn't Apply to VQVAE:**

VQVAE is a **deterministic model** (not probabilistic):
- Encoder outputs deterministic embeddings (no `mu` or `log_var`)
- No sampling: Uses vector quantization instead
- Latent codes are discrete (codebook indices), not continuous distributions
- **No probability distribution** over latent codes

**VQ Loss replaces KL** as the regularization term:
- Commitment loss: Encourages encoder to output embeddings close to codebook entries
- Quantization loss: Updates codebook entries to better represent encoder outputs
- Loss: `Loss = Recon_Loss + VQ_Loss`

**Architecture:**
```
VQVAE (Deterministic):
Input → Encoder → Embeddings → Quantize (find closest codebook entry) → Decoder → Output
                              ↓
                         VQ Loss (commitment + quantization)
```

### Generation Process

VQVAE generates samples by:
1. Sampling random embeddings from the codebook
2. Decoding those embeddings to generate images
3. Evaluating self-reconstruction quality

---

## VAEGAN Metrics

### Overview

VAEGAN (Variational Autoencoder Generative Adversarial Network) combines probabilistic VAE with adversarial training, resulting in unique loss components and evaluation metrics.

### Metrics Computed

1. **MSE** (Mean Squared Error)
   - Pixel-wise reconstruction loss
   - Computed for comparison purposes (VAEGAN uses feature-space MSE)

2. **BCE** (Binary Cross-Entropy)
   - Computed for comparison purposes
   - **Note**: VAEGAN uses feature-space MSE, not pixel-wise BCE or MSE

3. **Recon_Loss** (Reconstruction Loss)
   - **VAEGAN's actual reconstruction loss**
   - MSE computed in discriminator feature space (not pixel space)
   - Uses intermediate discriminator layer features for perceptual reconstruction
   - Lower is better

4. **KL** (Kullback-Leibler Divergence)
   - Computed directly from encoder outputs (`mu`, `log_var`)
   - Measures divergence between posterior q(z|x) and prior p(z) = N(0, I)
   - Same as in BetaVAE (VAEGAN is probabilistic)
   - Lower indicates better regularization

5. **ELBO** (Evidence Lower Bound)
   - Computed as `ELBO = MSE + KL` for comparison with BetaVAE
   - **Note**: VAEGAN's actual training uses feature-space recon_loss, not pixel MSE

6. **Encoder_Loss**
   - Encoder component: `Encoder_Loss = KL + Recon_Loss`
   - Combines regularization (KL) and reconstruction (feature-space MSE)

7. **Decoder_Loss**
   - Decoder component with adversarial component
   - Formula: `Decoder_Loss = (1 - α) * Recon_Loss - α * Discriminator_Loss`
   - Where α is the adversarial loss scale
   - Balances reconstruction quality and adversarial training

8. **Discriminator_Loss**
   - Adversarial loss from discriminator
   - Measures discriminator's ability to distinguish real vs generated images
   - Lower indicates better generation quality (discriminator is fooled)

9. **Total_Loss**
   - Sum of encoder_loss + decoder_loss + discriminator_loss
   - The overall training objective

10. **LPIPS, SSIM, GMSD**
    - Perceptual metrics (same as other models)

### Why Feature-Space Reconstruction Loss?

VAEGAN uses feature-space MSE instead of pixel-wise MSE because:

1. **Perceptual Quality**: Discriminator features capture high-level perceptual information
2. **Adversarial Training**: Aligns with GAN training paradigm
3. **Better Reconstructions**: Encourages perceptually similar images, not just pixel-accurate ones
4. **Stability**: Feature-space losses are often more stable than pixel losses

**Code Reference:**
```python
# VAEGAN loss function (from pythae/src/pythae/models/vae_gan/vae_gan_model.py)
# Feature maps of true data
true_discr_layer = self.discriminator(
    x, output_layer_levels=[self.reconstruction_layer]
)[f"embedding_layer_{self.reconstruction_layer}"]

# Feature maps of recon data
recon_discr_layer = self.discriminator(
    recon_x, output_layer_levels=[self.reconstruction_layer]
)[f"embedding_layer_{self.reconstruction_layer}"]

# MSE in feature space
recon_loss = 0.5 * F.mse_loss(
    true_discr_layer.reshape(N, -1),
    recon_discr_layer.reshape(N, -1),
    reduction="none",
).sum(dim=-1)
```

### Generation Process

VAEGAN generates samples by:
1. Sampling `z ~ N(0, I)` from standard normal prior
2. Decoding: `x = Decoder(z)`
3. Discriminator provides additional training signal

### Training Dynamics

VAEGAN uses **coupled optimizers** with conditional updates:

- **Encoder**: Always updated
- **Decoder**: Updated unless discriminator is too weak
- **Discriminator**: Updated unless it's too strong

This ensures balanced training between generator and discriminator.

---

## Model Architecture Comparison

### Side-by-Side Comparison

| Aspect | BetaVAE | VQVAE | VAEGAN |
|--------|---------|-------|--------|
| **Model Type** | Probabilistic | Deterministic | Probabilistic + Adversarial |
| **Latent Space** | Continuous distribution | Discrete codebook | Continuous distribution |
| **Encoder Output** | `mu`, `log_var` | Deterministic embeddings | `mu`, `log_var` |
| **Sampling** | Reparameterization trick | Vector quantization | Reparameterization trick |
| **Reconstruction Loss** | Pixel MSE/BCE (configurable) | Pixel MSE (hardcoded) | **Feature-space MSE** |
| **Regularization** | KL divergence | VQ loss | KL + Adversarial |
| **Discriminator** | No | No | Yes |
| **Generation** | `z ~ N(0, I)` | Codebook sampling | `z ~ N(0, I)` |
| **Training** | Single optimizer | Single optimizer | **Coupled optimizers** |
| **KL Divergence** | ✅ Yes | ❌ No (not applicable) | ✅ Yes |
| **VQ Loss** | ❌ No | ✅ Yes | ❌ No |
| **Adversarial Loss** | ❌ No | ❌ No | ✅ Yes |

### Architectural Diagrams

**BetaVAE (Probabilistic):**
```
Input → Encoder → [mu, log_var] → Sample z ~ N(mu, σ²) → Decoder → Output
                              ↓
                         KL(q(z|x)||p(z))
Loss = Recon_Loss + β * KL
```

**VQVAE (Deterministic):**
```
Input → Encoder → Embeddings → Quantize (codebook) → Decoder → Output
                              ↓
                         VQ Loss (commitment + quantization)
Loss = Recon_Loss + VQ_Loss
```

**VAEGAN (Probabilistic + Adversarial):**
```
Input → Encoder → [mu, log_var] → Sample z ~ N(mu, σ²) → Decoder → Output
                              ↓                              ↓
                         KL(q(z|x)||p(z))              Discriminator
                                                              ↓
                                                    Feature-space MSE
Loss = Encoder_Loss + Decoder_Loss + Discriminator_Loss
where:
  Encoder_Loss = KL + Recon_Loss (feature-space)
  Decoder_Loss = (1-α) * Recon_Loss - α * Discriminator_Loss
  Discriminator_Loss = BCE(real) + BCE(fake)
```

### Understanding Loss Components

#### BetaVAE Loss
```
Loss = Recon_Loss + β * KL
```
- **Recon_Loss**: Pixel-wise MSE or BCE
- **KL**: Regularizes latent distribution to match prior N(0, I)
- **β**: Controls trade-off between reconstruction and regularization

#### VQVAE Loss
```
Loss = Recon_Loss + VQ_Loss
```
- **Recon_Loss**: Pixel-wise MSE (hardcoded)
- **VQ_Loss**: Commitment loss + Quantization loss
- No probabilistic component

#### VAEGAN Loss
```
Total_Loss = Encoder_Loss + Decoder_Loss + Discriminator_Loss

Encoder_Loss = KL + Recon_Loss (feature-space)
Decoder_Loss = (1 - α) * Recon_Loss - α * Discriminator_Loss
Discriminator_Loss = BCE(real_images, 1) + BCE(generated_images, 0)
```
- **Encoder_Loss**: Combines KL regularization and feature-space reconstruction
- **Decoder_Loss**: Balances reconstruction and adversarial training
- **Discriminator_Loss**: Trains discriminator to distinguish real from fake
- **α**: Adversarial loss scale parameter

### Model Selection Guide

**Use BetaVAE if:**
- You need probabilistic latent space and sampling
- You want simple pixel-wise reconstruction loss
- You need interpretable KL divergence regularization

**Use VQVAE if:**
- You need discrete latent representations
- You want codebook learning for discrete codes
- You prefer deterministic encoding

**Use VAEGAN if:**
- You need high-quality generation with adversarial training
- You want perceptually better reconstructions (feature-space loss)
- You can handle more complex training dynamics (coupled optimizers)

### Metric Interpretation

**Reconstruction Quality:**
- **MSE**: Lower is better (pixel accuracy)
- **LPIPS**: Lower is better (perceptual similarity)
- **SSIM**: Higher is better (structural similarity)
- **GMSD**: Lower is better (gradient similarity)

**Regularization:**
- **KL Divergence** (BetaVAE, VAEGAN): Lower indicates better regularization, but too low may indicate posterior collapse
- **VQ Loss** (VQVAE): Lower indicates better codebook utilization

**Generation Quality:**
- **Discriminator Loss** (VAEGAN): Lower indicates better generation (discriminator is fooled)
- **Self-reconstruction metrics**: Measure consistency of generated samples

---

## References

### Papers

- **VQVAE**: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (van den Oord et al., 2017)
- **BetaVAE**: [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) (Higgins et al., 2017)
- **VAEGAN**: [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300) (Larsen et al., 2016)

### Implementation

- **PyTorch VAE Library**: [pythae](https://github.com/clementchadebec/benchmark_VAE) - Comprehensive VAE implementations
- **Hugging Face Hub**: [clementchadebec models](https://huggingface.co/models?search=clementchadebec) - Pretrained model repository

### Metrics Libraries

- **LPIPS**: [Perceptual Similarity](https://github.com/richzhang/PerceptualSimilarity)
- **PIQ**: [PyTorch Image Quality](https://github.com/photosynthesis-team/piq)
- **Weights & Biases**: [Experiment Tracking](https://wandb.ai/)

---

## Additional Resources

- **Project README**: See `README.md` for project overview and usage instructions
- **Training Scripts**: See `training_shell_scripts/` for GPU/CPU training automation
- **Evaluation Scripts**: See `evaluate_vqvae.py` and `evaluate_vaegan.py` for detailed evaluation
- **Configuration Files**: See `configs/celeba/` for model and training configurations

