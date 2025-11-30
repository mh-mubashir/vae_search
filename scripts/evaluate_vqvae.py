#!/usr/bin/env python3
"""
Evaluation script for VQVAE models.
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
from torchvision.datasets import CelebA
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from pythae.models import AutoModel
from pythae.data.datasets import BaseDataset


def load_eval_data(data_root="data", n_samples=10000):
    """Load CelebA evaluation data."""
    transform = Compose([
        CenterCrop(178),
        Resize((32, 32)),
        ToTensor(),
    ])
    
    celeba_eval = CelebA(
        root=data_root,
        split="valid",  # Use validation split for evaluation
        download=False,
        transform=transform
    )
    
    n_samples = min(n_samples, len(celeba_eval))
    xs = []
    for i in range(n_samples):
        img, _ = celeba_eval[i]
        xs.append(img)
    x_eval = torch.stack(xs, dim=0)
    
    return x_eval


@torch.no_grad()
def compute_recon_metrics_from_out(x, out, device):
    """
    Compute reconstruction metrics for VQVAE.
    
    Args:
        x: [B, C, H, W] ground-truth images in [0,1]
        out: model output with keys: recon_x, recon_loss, vq_loss, loss
        device: torch device
    """
    x_hat = out.recon_x  # [B, C, H, W]
    
    # MSE (reconstruction loss)
    mse = F.mse_loss(x_hat, x, reduction="mean").item()
    
    # VQ Loss (vector quantization loss)
    vq_loss = out.vq_loss.mean().item() if hasattr(out.vq_loss, 'mean') else out.vq_loss.item()
    
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
        "VQ_Loss": vq_loss,
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
        metrics = compute_recon_metrics_from_out(batch, out, device)
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
    
    1) Sample random embeddings from codebook
    2) Decode z -> x_gen
    3) Pass x_gen back through the full model (encoder + decoder) to get x_gen_hat
    4) Compute same metrics between x_gen and x_gen_hat
    """
    torch.manual_seed(seed)
    model.eval()
    
    # Get codebook embeddings
    if hasattr(model.quantizer, 'embeddings'):
        if isinstance(model.quantizer.embeddings, torch.nn.Parameter):
            codebook = model.quantizer.embeddings.data
        elif isinstance(model.quantizer.embeddings, torch.Tensor):
            codebook = model.quantizer.embeddings
        else:
            # For QuantizerEMA, embeddings is a buffer
            codebook = model.quantizer.embeddings
    else:
        raise ValueError("Could not find codebook embeddings")
    
    num_embeddings = codebook.shape[0]
    embedding_dim = codebook.shape[1]
    
    # Step 1: Determine spatial dimensions by encoding a dummy image
    dummy_input = torch.randn(1, *model.model_config.input_dim, device=device)
    with torch.no_grad():
        encoder_out = model.encoder(dummy_input)
        embeddings_shape = encoder_out.embedding.shape
    
    # Step 2: Sample random quantized embeddings matching the encoder output shape
    if len(embeddings_shape) == 2:
        # Flatten case: encoder outputs [B, latent_dim]
        # Sample random embeddings and reshape to match decoder input
        random_indices = torch.randint(0, num_embeddings, (num_samples,), device=device)
        z_quant = codebook[random_indices]  # [num_samples, embedding_dim]
        # Decoder expects flattened input
    else:
        # Spatial case: encoder outputs [B, C, H, W] 
        h, w = embeddings_shape[2], embeddings_shape[3]
        # Sample random indices for each spatial location
        num_spatial = h * w
        random_indices = torch.randint(0, num_embeddings, 
                                      (num_samples, num_spatial), device=device)
        # Get embeddings: [num_samples, num_spatial, embedding_dim]
        z_spatial = codebook[random_indices]
        # Reshape to match encoder output: [num_samples, embedding_dim, h, w]
        z_spatial = z_spatial.reshape(num_samples, h, w, embedding_dim)
        z_spatial = z_spatial.permute(0, 3, 1, 2)  # [num_samples, embedding_dim, h, w]
        # Flatten for decoder (matching the forward pass logic)
        z_quant = z_spatial.reshape(num_samples, -1)
    
    # Step 3: Decode
    dec_out = model.decoder(z_quant)
    x_gen = dec_out.reconstruction.clamp(0.0, 1.0)
    
    # Step 4: Pass generated images through encoder+decoder
    out_gen = model({"data": x_gen})
    
    # Step 5: Metrics x_gen vs recon(x_gen)
    metrics_gen = compute_recon_metrics_from_out(x_gen, out_gen, device)
    return metrics_gen


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQVAE model")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint folder (e.g., 'my_models_on_celeba/VQVAE_training_YYYY-MM-DD_HH-MM-SS/final_model')"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory for CelebA dataset"
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
        default="vqvae-evaluation",
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
    x_eval = load_eval_data(data_root=args.data_root, n_samples=args.n_eval_samples)
    print(f"Loaded {len(x_eval)} evaluation samples")
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or os.path.basename(args.checkpoint_path),
            config={
                "checkpoint_path": args.checkpoint_path,
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

python evaluate_vqvae.py \
    --checkpoint_path "/home/mubashir.m/ondemand/data/sys/dashboard/batch_connect/sys/desktop-native-courses/output/e6d95e33-bb4f-4913-a299-51074c8c526e/testing/vae_search/scripts/my_models_on_celeba/VQVAE_training_2025-11-29_21-31-53/checkpoint_epoch_17" \
    --data_root "data" \
    --n_eval_samples 10000 \
    --n_gen_samples 64 \
    --batch_size 64 \
    --use_wandb \
    --wandb_project "vqvae-evaluation"