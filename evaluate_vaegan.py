#!/usr/bin/env python3
"""
Evaluation script for VAEGAN models.
Computes reconstruction and generation metrics similar to BetaVAE evaluation.
Logs metrics to wandb.
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import lpips
import piq
import wandb
from pythae.models import AutoModel
from pythae.data.datasets import BaseDataset


def load_eval_data(data_root="data", dataset="celeba", n_samples=None):
    """
    Load evaluation data from .npz file (same format as training script).
    
    Args:
        data_root: Root directory containing the dataset folder
        dataset: Dataset name (default: "celeba")
        n_samples: Number of samples to use (None = use all)
    
    Returns:
        x_eval: torch.Tensor of shape [N, C, H, W] in range [0, 1]
    """
    # Get the script directory (same as training.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct path to eval_data.npz
    eval_data_path = os.path.join(script_dir, data_root, dataset, "eval_data.npz")
    
    if not os.path.exists(eval_data_path):
        raise FileNotFoundError(
            f"Evaluation data not found at: {eval_data_path}\n"
            f"Please ensure 'eval_data.npz' exists in 'data/{dataset}/' folder."
        )
    
    # Load data (expected to be in range [0-255] with channel in first position)
    eval_data = np.load(eval_data_path)["data"] / 255.0
    
    # Convert to torch tensor
    x_eval = torch.from_numpy(eval_data).float()
    
    # Limit number of samples if specified
    if n_samples is not None and n_samples < len(x_eval):
        x_eval = x_eval[:n_samples]
    
    print(f"Loaded {len(x_eval)} evaluation samples from {eval_data_path}")
    print(f"Data shape: {x_eval.shape}, Range: [{x_eval.min():.3f}, {x_eval.max():.3f}]")
    
    return x_eval


@torch.no_grad()
def compute_recon_metrics_from_out(x, out, model, device):
    """
    Compute reconstruction metrics for VAEGAN.
    
    Args:
        x: [B, C, H, W] ground-truth images in [0,1]
        out: model output with keys: recon_x, recon_loss, encoder_loss, decoder_loss, discriminator_loss
        model: VAEGAN model (needed to compute KL from encoder outputs)
        device: torch device
    
    Note: VAEGAN is a probabilistic VAE with adversarial training:
    - Has KL divergence (like BetaVAE)
    - Uses MSE in discriminator feature space for reconstruction
    - Has discriminator loss for adversarial training
    - Can use BCE or MSE for reconstruction (but uses feature-space MSE)
    """
    x_hat = out.recon_x  # [B, C, H, W]
    
    # MSE (pixel-wise reconstruction loss)
    mse = F.mse_loss(x_hat, x, reduction="mean").item()
    
    # BCE (for comparison, though VAEGAN uses feature-space MSE)
    bce = F.binary_cross_entropy(
        x_hat.clamp(0.0, 1.0).reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="mean"
    ).item()
    
    # Reconstruction loss (from model - MSE in discriminator feature space)
    recon_loss = out.recon_loss.mean().item() if hasattr(out.recon_loss, 'mean') else out.recon_loss.item()
    
    # Compute KL divergence directly from encoder outputs
    encoder_output = model.encoder(x)
    mu = encoder_output.embedding
    log_var = encoder_output.log_covariance
    kl = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)).mean().item()
    
    # ELBO = Recon_Loss + KL (for comparison with BetaVAE)
    # Note: VAEGAN uses feature-space recon_loss, but we compute ELBO with pixel MSE for comparison
    elbo = mse + kl
    
    # Encoder loss
    encoder_loss = out.encoder_loss.mean().item() if hasattr(out.encoder_loss, 'mean') else out.encoder_loss.item()
    
    # Decoder loss
    decoder_loss = out.decoder_loss.mean().item() if hasattr(out.decoder_loss, 'mean') else out.decoder_loss.item()
    
    # Discriminator loss
    discriminator_loss = out.discriminator_loss.mean().item() if hasattr(out.discriminator_loss, 'mean') else out.discriminator_loss.item()
    
    # Total Loss
    total_loss = out.loss.mean().item() if hasattr(out.loss, 'mean') else out.loss.item()
    
    # Perceptual metrics expect 3 channels
    if x.shape[1] == 1:
        x_lp = x.repeat(1, 3, 1, 1)
        x_hat_lp = x_hat.repeat(1, 3, 1, 1)
    else:
        x_lp = x
        x_hat_lp = x_hat
    
    # LPIPS (average over batch)
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    lp = lpips_fn(x_lp, x_hat_lp).mean().item()
    
    # SSIM, GMSD from piq
    ssim_val = piq.ssim(x_hat, x, data_range=1.0).item()
    gmsd_val = piq.gmsd(x_hat, x, data_range=1.0).item()
    
    return {
        "MSE": mse,
        "BCE": bce,
        "Recon_Loss": recon_loss,  # Feature-space MSE from discriminator
        "KL": kl,  # KL divergence computed from encoder outputs
        "ELBO": elbo,  # Evidence Lower Bound = MSE + KL (for comparison)
        "Encoder_Loss": encoder_loss,
        "Decoder_Loss": decoder_loss,
        "Discriminator_Loss": discriminator_loss,
        "Total_Loss": total_loss,
        "LPIPS": lp,
        "SSIM": ssim_val,
        "GMSD": gmsd_val,
    }


@torch.no_grad()
def eval_reconstruction_metrics(model, x_eval, device, batch_size=64):
    """Evaluate reconstruction metrics on evaluation set."""
    model.eval()
    all_metrics = []
    
    # Process in batches
    for i in range(0, len(x_eval), batch_size):
        batch = x_eval[i:i+batch_size].to(device)
        out = model({"data": batch})
        metrics = compute_recon_metrics_from_out(batch, out, model, device)
        all_metrics.append(metrics)
    
    # Average over all batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics


@torch.no_grad()
def eval_generation_self_recon_metrics(model, num_samples=64, seed=999, device=None):
    """
    Evaluate generation metrics using self-reconstruction.
    
    1) Sample z ~ N(0, I) from prior
    2) Decode z -> x_gen
    3) Pass x_gen back through the full model (encoder + decoder) to get x_gen_hat
    4) Compute same metrics between x_gen and x_gen_hat
    """
    torch.manual_seed(seed)
    model.eval()
    
    latent_dim = model.model_config.latent_dim
    
    # Step 1: Sample z from prior N(0, I)
    z_prior = torch.randn(num_samples, latent_dim, device=device)
    
    # Step 2: Decode
    dec_out = model.decoder(z_prior)
    x_gen = dec_out.reconstruction.clamp(0.0, 1.0)
    
    # Step 3: Pass generated images through encoder+decoder
    out_gen = model({"data": x_gen})
    
    # Step 4: Metrics x_gen vs recon(x_gen)
    metrics_gen = compute_recon_metrics_from_out(x_gen, out_gen, model, device)
    return metrics_gen


def main():
    parser = argparse.ArgumentParser(description="Evaluate VAEGAN model")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint folder (e.g., 'my_models_on_celeba/VAEGAN_training_YYYY-MM-DD_HH-MM-SS/final_model')"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory containing dataset folder"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="celeba",
        help="Dataset name (default: celeba)"
    )
    parser.add_argument(
        "--n_eval_samples",
        type=int,
        default=10000,
        help="Number of evaluation samples"
    )
    parser.add_argument(
        "--n_gen_samples",
        type=int,
        default=64,
        help="Number of generated samples for generation metrics"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log metrics to wandb"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="vaegan-evaluation",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity name (optional)"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (optional)"
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {args.checkpoint_path}")
    
    model = AutoModel.load_from_folder(args.checkpoint_path)
    model.to(device)
    model.eval()
    print(f"Model loaded: {model.model_name}")
    
    # Load evaluation data
    print("Loading evaluation data...")
    x_eval = load_eval_data(
        data_root=args.data_root, 
        dataset=args.dataset,
        n_samples=args.n_eval_samples
    )
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or os.path.basename(args.checkpoint_path),
            config={
                "checkpoint_path": args.checkpoint_path,
                "dataset": args.dataset,
                "n_eval_samples": args.n_eval_samples,
                "n_gen_samples": args.n_gen_samples,
                "batch_size": args.batch_size,
            }
        )
    
    # Evaluate reconstruction metrics
    print("\n=== Evaluating Reconstruction Metrics ===")
    recon_metrics = eval_reconstruction_metrics(
        model, x_eval, device, batch_size=args.batch_size
    )
    print("Reconstruction Metrics:")
    for key, value in recon_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Evaluate generation metrics
    print("\n=== Evaluating Generation Metrics ===")
    gen_metrics = eval_generation_self_recon_metrics(
        model, num_samples=args.n_gen_samples, seed=999, device=device
    )
    print("Generation Metrics:")
    for key, value in gen_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Log to wandb
    if args.use_wandb:
        # Log reconstruction metrics with prefix
        recon_wandb = {f"recon/{k}": v for k, v in recon_metrics.items()}
        # Log generation metrics with prefix
        gen_wandb = {f"gen/{k}": v for k, v in gen_metrics.items()}
        wandb.log({**recon_wandb, **gen_wandb})
        print("\nMetrics logged to wandb!")
    
    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()

