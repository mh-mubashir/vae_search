#!/usr/bin/env python3
"""
Script to test pretrained VAEGAN model on CelebA dataset.

This script:
1. Loads a pretrained VAEGAN model from HuggingFace Hub or local path
2. Loads CelebA dataset
3. Tests reconstruction capabilities
4. Tests generation capabilities using various samplers
5. Saves visualization results

Usage:
    python test_vaegan_celeba.py --model_path clementchadebec/vaegan_celeba --output_dir results
    python test_vaegan_celeba.py --model_path ./my_models/VAEGAN_training_2024-01-01_12-00-00/final_model --output_dir results
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pythae.models import AutoModel, VAEGAN, VAEGANConfig
from pythae.models.nn.benchmarks.celeba import (
    Decoder_Conv_AE_CELEBA,
    Discriminator_Conv_CELEBA,
    Encoder_Conv_VAE_CELEBA,
)
from pythae.samplers import (
    GaussianMixtureSampler,
    GaussianMixtureSamplerConfig,
    NormalSampler,
)

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

# Set up matplotlib backend
plt.switch_backend("Agg")


def load_celeba_data(data_path: str, num_samples: int = None):
    """Load CelebA dataset from npz files.

    Args:
        data_path: Path to directory containing train_data.npz and eval_data.npz
        num_samples: Number of samples to use (None for all)

    Returns:
        Tuple of (train_data, eval_data) as numpy arrays normalized to [0, 1]
    """
    train_path = os.path.join(data_path, "train_data.npz")
    eval_path = os.path.join(data_path, "eval_data.npz")

    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        raise FileNotFoundError(
            f"CelebA data not found at {data_path}. "
            "Please ensure train_data.npz and eval_data.npz exist. "
            "You can download CelebA using: python data-download.py celeba"
        )

    logger.info(f"Loading CelebA data from {data_path}...")
    train_data = np.load(train_path)["data"] / 255.0
    eval_data = np.load(eval_path)["data"] / 255.0

    if num_samples is not None:
        train_data = train_data[:num_samples]
        eval_data = eval_data[:num_samples]

    logger.info(f"Loaded train data: {train_data.shape}, eval data: {eval_data.shape}")
    return train_data, eval_data


def load_model(model_path: str, device: str = "cuda"):
    """Load VAEGAN model from HuggingFace Hub or local path.

    Args:
        model_path: HuggingFace Hub path (e.g., 'username/model_name') or local path
        device: Device to load model on

    Returns:
        Loaded VAEGAN model
    """
    logger.info(f"Loading model from {model_path}...")

    # Check if it's a local path
    if os.path.exists(model_path):
        try:
            model = AutoModel.load_from_folder(model_path)
            logger.info(f"Successfully loaded model from local path: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from local path: {e}")
            raise
    else:
        # Try loading from HuggingFace Hub
        try:
            model = AutoModel.load_from_hf_hub(model_path)
            logger.info(f"Successfully loaded model from HuggingFace Hub: {model_path}")
        except Exception as e:
            logger.error(
                f"Failed to load model from HuggingFace Hub: {e}\n"
                "Please ensure:\n"
                "1. The model path is correct (format: 'username/model_name')\n"
                "2. You have internet connection\n"
                "3. huggingface_hub is installed: pip install huggingface_hub\n"
                "4. You're logged in: huggingface-cli login (if model is private)\n"
                "\n"
                "Alternatively, you can train a model first using:\n"
                "python training.py --dataset celeba --model_name vaegan"
            )
            raise

    model = model.to(device)
    model.eval()
    return model


def test_reconstruction(model, eval_data, device, num_samples=25, output_dir=None):
    """Test reconstruction capabilities of the model.

    Args:
        model: VAEGAN model
        eval_data: Evaluation data
        device: Device to run on
        num_samples: Number of samples to reconstruct
        output_dir: Directory to save results

    Returns:
        Tuple of (original_images, reconstructed_images)
    """
    logger.info(f"Testing reconstruction on {num_samples} samples...")

    # Select samples
    test_samples = torch.tensor(eval_data[:num_samples], dtype=torch.float32).to(device)

    # Reconstruct
    with torch.no_grad():
        reconstructions = model.reconstruct(test_samples)

    original_images = test_samples.cpu().numpy()
    reconstructed_images = reconstructions.cpu().numpy()

    # Visualize
    if output_dir:
        save_reconstruction_visualization(
            original_images, reconstructed_images, output_dir
        )

    logger.info("Reconstruction test completed!")
    return original_images, reconstructed_images


def save_reconstruction_visualization(
    original_images, reconstructed_images, output_dir, grid_size=5
):
    """Save reconstruction visualization as image grid.

    Args:
        original_images: Original images array
        reconstructed_images: Reconstructed images array
        output_dir: Directory to save visualization
        grid_size: Size of the grid (grid_size x grid_size)
    """
    num_samples = min(len(original_images), grid_size * grid_size)
    fig, axes = plt.subplots(
        2 * grid_size, grid_size, figsize=(grid_size * 2, grid_size * 4)
    )

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx >= num_samples:
                break

            # Original image
            orig_img = original_images[idx].transpose(1, 2, 0)
            axes[2 * i, j].imshow(np.clip(orig_img, 0, 1))
            axes[2 * i, j].set_title(f"Original {idx}")
            axes[2 * i, j].axis("off")

            # Reconstructed image
            recon_img = reconstructed_images[idx].transpose(1, 2, 0)
            axes[2 * i + 1, j].imshow(np.clip(recon_img, 0, 1))
            axes[2 * i + 1, j].set_title(f"Reconstructed {idx}")
            axes[2 * i + 1, j].axis("off")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "reconstruction_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved reconstruction visualization to {output_path}")


def test_generation(model, train_data, device, num_samples=25, output_dir=None):
    """Test generation capabilities using various samplers.

    Args:
        model: VAEGAN model
        train_data: Training data (for fitting samplers that need it)
        device: Device to run on
        num_samples: Number of samples to generate
        output_dir: Directory to save results

    Returns:
        Dictionary of sampler names and generated samples
    """
    logger.info(f"Testing generation with {num_samples} samples...")
    results = {}

    # 1. Normal Sampler
    logger.info("Testing NormalSampler...")
    normal_sampler = NormalSampler(model=model)
    with torch.no_grad():
        gen_normal = normal_sampler.sample(num_samples=num_samples, return_gen=True)
    results["normal"] = gen_normal.cpu().numpy()
    logger.info("NormalSampler completed!")

    # 2. Gaussian Mixture Sampler
    logger.info("Testing GaussianMixtureSampler...")
    try:
        gmm_config = GaussianMixtureSamplerConfig(n_components=10)
        gmm_sampler = GaussianMixtureSampler(
            sampler_config=gmm_config, model=model
        )
        # Fit the sampler on training data
        train_tensor = torch.tensor(train_data[:1000], dtype=torch.float32).to(device)
        gmm_sampler.fit(train_tensor)
        # Generate samples
        with torch.no_grad():
            gen_gmm = gmm_sampler.sample(num_samples=num_samples, return_gen=True)
        results["gaussian_mixture"] = gen_gmm.cpu().numpy()
        logger.info("GaussianMixtureSampler completed!")
    except Exception as e:
        logger.warning(f"GaussianMixtureSampler failed: {e}")

    # Visualize results
    if output_dir:
        save_generation_visualization(results, output_dir)

    logger.info("Generation test completed!")
    return results


def save_generation_visualization(results, output_dir, grid_size=5):
    """Save generation visualization as image grids.

    Args:
        results: Dictionary of sampler names and generated samples
        output_dir: Directory to save visualizations
        grid_size: Size of the grid (grid_size x grid_size)
    """
    for sampler_name, generated_images in results.items():
        num_samples = min(len(generated_images), grid_size * grid_size)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))

        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx >= num_samples:
                    axes[i, j].axis("off")
                    continue

                img = generated_images[idx].transpose(1, 2, 0)
                axes[i, j].imshow(np.clip(img, 0, 1))
                axes[i, j].set_title(f"Sample {idx}")
                axes[i, j].axis("off")

        plt.suptitle(f"Generated Samples - {sampler_name.replace('_', ' ').title()}", fontsize=14)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"generation_{sampler_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {sampler_name} generation visualization to {output_path}")


def test_interpolation(model, eval_data, device, num_pairs=5, granularity=10, output_dir=None):
    """Test interpolation between pairs of images.

    Args:
        model: VAEGAN model
        eval_data: Evaluation data
        device: Device to run on
        num_pairs: Number of image pairs to interpolate
        granularity: Number of interpolation steps
        output_dir: Directory to save results
    """
    logger.info(f"Testing interpolation with {num_pairs} pairs and {granularity} steps...")

    # Select pairs
    pairs = []
    for i in range(num_pairs):
        idx1 = i * 2
        idx2 = i * 2 + 1
        if idx2 >= len(eval_data):
            break
        pairs.append((idx1, idx2))

    all_interpolations = []
    for idx1, idx2 in pairs:
        img1 = torch.tensor(eval_data[idx1:idx1+1], dtype=torch.float32).to(device)
        img2 = torch.tensor(eval_data[idx2:idx2+1], dtype=torch.float32).to(device)

        with torch.no_grad():
            interpolations = model.interpolate(img1, img2, granularity=granularity)

        all_interpolations.append(interpolations.cpu().numpy())

    # Visualize
    if output_dir:
        save_interpolation_visualization(all_interpolations, output_dir)

    logger.info("Interpolation test completed!")
    return all_interpolations


def save_interpolation_visualization(interpolations, output_dir):
    """Save interpolation visualization.

    Args:
        interpolations: List of interpolation arrays
        output_dir: Directory to save visualization
    """
    num_pairs = len(interpolations)
    granularity = interpolations[0].shape[1] if interpolations else 10

    fig, axes = plt.subplots(num_pairs, granularity, figsize=(granularity * 1.5, num_pairs * 1.5))

    if num_pairs == 1:
        axes = axes.reshape(1, -1)

    for i, interp in enumerate(interpolations):
        for j in range(granularity):
            img = interp[0, j].transpose(1, 2, 0)
            axes[i, j].imshow(np.clip(img, 0, 1))
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title("Start", fontsize=8)
            elif j == granularity - 1:
                axes[i, j].set_title("End", fontsize=8)

    plt.suptitle("Latent Space Interpolations", fontsize=14)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "interpolation_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved interpolation visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test pretrained VAEGAN model on CelebA dataset"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="clementchadebec/vaegan_celeba",
        help="Path to pretrained model. Can be HuggingFace Hub path (e.g., 'username/model_name') "
        "or local path to model directory",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to CelebA data directory containing train_data.npz and eval_data.npz. "
        "If not provided, will look in 'data/celeba' relative to script location",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="vaegan_celeba_results",
        help="Directory to save test results and visualizations",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=25,
        help="Number of samples to use for reconstruction and generation tests",
    )

    parser.add_argument(
        "--num_interpolation_pairs",
        type=int,
        default=5,
        help="Number of image pairs for interpolation test",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )

    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation tests (faster)",
    )

    parser.add_argument(
        "--skip_interpolation",
        action="store_true",
        help="Skip interpolation tests",
    )

    args = parser.parse_args()

    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_path is None:
        args.data_path = os.path.join(script_dir, "data", "celeba")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("VAEGAN CelebA Testing Script")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    try:
        # Load model
        model = load_model(args.model_path, device=args.device)
        logger.info(f"Model loaded successfully! Model type: {model.model_name}")

        # Load data
        train_data, eval_data = load_celeba_data(args.data_path, num_samples=None)

        # Test reconstruction
        logger.info("\n" + "=" * 80)
        logger.info("TESTING RECONSTRUCTION")
        logger.info("=" * 80)
        test_reconstruction(
            model, eval_data, args.device, num_samples=args.num_samples, output_dir=args.output_dir
        )

        # Test generation
        if not args.skip_generation:
            logger.info("\n" + "=" * 80)
            logger.info("TESTING GENERATION")
            logger.info("=" * 80)
            test_generation(
                model,
                train_data,
                args.device,
                num_samples=args.num_samples,
                output_dir=args.output_dir,
            )

        # Test interpolation
        if not args.skip_interpolation:
            logger.info("\n" + "=" * 80)
            logger.info("TESTING INTERPOLATION")
            logger.info("=" * 80)
            test_interpolation(
                model,
                eval_data,
                args.device,
                num_pairs=args.num_interpolation_pairs,
                output_dir=args.output_dir,
            )

        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


