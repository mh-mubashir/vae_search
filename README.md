# VAE Architecture Evaluation Project

This repository serves as the project repository for the **Machine Learning for Small Data** course project.

## Team Members

- **Hamza Muabshir**
- **Alexsei Krotov**

## Project Overview

The goal of this project is to evaluate three implementations of the Variational Autoencoder (VAE) architecture. This includes:

- **Training** the models on various datasets
- **Evaluating** the models post-training
- **Experimenting** with different model parameters and dynamics to identify optimal configurations
- **Comparing** results between the three VAE implementations
- **Documenting** implementation challenges and solutions
- **Providing** a complete implementation framework for reproducibility

## Objectives

1. **Model Evaluation**: Train and evaluate three different VAE architecture implementations
2. **Parameter Optimization**: Experiment with different hyperparameters and model configurations
3. **Comparative Analysis**: Compare results, performance metrics, and implementation challenges across models
4. **Reproducibility**: Provide clear documentation and scripts for easy replication
5. **Model Sharing**: Upload trained models to Hugging Face for easy access and testing

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended) or CPU
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd vae_search
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in editable mode:
```bash
pip install -e .
```

## Quick Start

### Training a Model

To train a VAE model, use the `training.py` script:

```bash
python scripts/training.py \
    --dataset mnist \
    --model_name vae \
    --model_config scripts/configs/mnist/vae_config.json \
    --training_config scripts/configs/mnist/base_training_config.json
```

**Available datasets**: `mnist`, `cifar10`, `celeba`

**Available models**: `vae`, `beta_vae`, `iwae`, `wae`, `info_vae`, `rae_gp`, `rae_l2`, `vamp`, `hvae`, `rhvae`, `aae`, `vaegan`, `vqvae`, and more.

### Testing a Model

After training, models are saved in the `my_models/` directory. You can test them using the provided test scripts:

```bash
# Test VAEGAN on CelebA
python scripts/test_vaegan_celeba.py \
    --model_path ./my_models/VAEGAN_training_YYYY-MM-DD_HH-MM-SS/final_model \
    --output_dir results

# Test VQ-VAE on CelebA
python scripts/test_vqvae_celeba.py \
    --model_path ./my_models/VQVAE_training_YYYY-MM-DD_HH-MM-SS/final_model \
    --output_dir results
```

## Project Structure

```
vae_search/
├── scripts/
│   ├── training.py              # Main training script
│   ├── configs/                 # Model and training configurations
│   │   ├── mnist/
│   │   ├── cifar10/
│   │   └── celeba/
│   ├── data/                    # Dataset directory (not tracked in git)
│   │   ├── mnist/
│   │   ├── cifar10/
│   │   └── celeba/
│   ├── test_vaegan_celeba.py    # VAEGAN testing script
│   └── test_vqvae_celeba.py     # VQ-VAE testing script
├── my_models/                   # Trained models directory (not tracked in git)
├── src/pythae/                  # Core library code
├── tests/                       # Test suite
└── README.md                    # This file
```

## Data Preparation

Datasets should be placed in `scripts/data/<dataset_name>/` with the following structure:

```
scripts/data/
└── <dataset_name>/
    ├── train_data.npz          # Training data
    └── eval_data.npz            # Evaluation data
```

The data files should be loadable as NumPy arrays with shape `(n_samples, channels, height, width)` and values in the range `[0, 255]`.

## Model Configurations

Model configurations are stored in JSON files under `scripts/configs/`. You can modify these files to experiment with different hyperparameters:

- **Model configurations**: Define architecture, latent dimensions, loss functions, etc.
- **Training configurations**: Define batch size, learning rate, number of epochs, optimizer settings, etc.

Example configuration files:
- `scripts/configs/mnist/vae_config.json`
- `scripts/configs/mnist/base_training_config.json`

## Hugging Face Integration

Trained models will be uploaded to Hugging Face Hub for easy access and sharing. To use a model from Hugging Face:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Test a model from Hugging Face Hub
python scripts/test_vaegan_celeba.py \
    --model_path <huggingface-username>/<model-name> \
    --output_dir results
```

If the model is private, authenticate first:
```bash
huggingface-cli login
```

## Evaluation and Comparison

Our evaluation framework includes:

1. **Reconstruction Quality**: Measure how well models reconstruct input images
2. **Generation Quality**: Evaluate the quality of generated samples
3. **Latent Space Analysis**: Examine the learned latent representations
4. **Interpolation**: Test smoothness of latent space interpolations
5. **Performance Metrics**: Track training time, memory usage, and convergence

### Running Evaluations

After training, models are automatically evaluated. Results are saved in the model directory. You can also run custom evaluations using the test scripts.

## Implementation Notes

### Challenges Encountered

During implementation, we encountered and resolved several challenges:

1. **Memory Management**: Large datasets (e.g., CelebA) require careful batch size tuning
2. **Hyperparameter Tuning**: Finding optimal learning rates and regularization weights
3. **Training Stability**: Some architectures require specific initialization or training schedules
4. **Hardware Constraints**: CPU vs GPU training considerations

### Solutions and Best Practices

- Use CPU-optimized configurations for systems without GPU
- Start with smaller batch sizes and gradually increase
- Monitor training metrics to detect convergence or instability
- Save checkpoints regularly to resume training if needed

## Contributing

This is a course project repository. For questions or issues, please contact the team members.

## License

See the LICENSE file for details.

## Acknowledgments

This project is built on top of the [pythae](https://github.com/clementchadebec/benchmark_VAE) library, which provides a unified framework for various VAE architectures.

## Citation

If you use this work, please cite:

```bibtex
@software{vae_search,
  title = {VAE Architecture Evaluation Project},
  author = {Muabshir, Hamza and Krotov, Alexsei},
  year = {2024},
  url = {<repository-url>}
}
```

