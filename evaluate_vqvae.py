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
def compute_recon_metrics_from_out(x, out, device):
    """
    Compute reconstruction metrics for VQVAE.
    
    Args:
        x: [B, C, H, W] ground-truth images in [0,1]
        out: model output with keys: recon_x, recon_loss, vq_loss, loss
        device: torch device
    
    Note: VQVAE uses MSE (not BCE) for reconstruction loss and VQ loss (not KL) for regularization.
    Unlike BetaVAE which has:
    - BCE: Binary cross-entropy reconstruction loss (optional, can use MSE instead)
    - KL: Kullback-Leibler divergence between posterior q(z|x) and prior p(z)
    VQVAE has:
    - MSE: Mean squared error reconstruction loss (hardcoded, no BCE option)
    - VQ_Loss: Vector quantization loss (commitment + quantization loss, replaces KL)
    """
    x_hat = out.recon_x  # [B, C, H, W]
    
    # MSE (reconstruction loss) - VQVAE always uses MSE, never BCE
    mse = F.mse_loss(x_hat, x, reduction="mean").item()
    
    # Optional: Compute BCE for comparison (even though VQVAE doesn't use it)
    # This allows comparison with BetaVAE metrics
    bce = F.binary_cross_entropy(
        x_hat.clamp(0.0, 1.0).reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="mean"
    ).item()
    
    # VQ Loss (vector quantization loss) - replaces KL divergence in VQVAE
    # This measures how well the codebook represents encoder outputs
    vq_loss = out.vq_loss.mean().item() if hasattr(out.vq_loss, 'mean') else out.vq_loss.item()
    
    # Total Loss
    total_loss = out.loss.mean().item() if hasattr(out.loss, 'mean') else out.loss.item()
    
    # Note: KL divergence doesn't exist for VQVAE since it's not a probabilistic model
    # VQVAE uses deterministic encoding + vector quantization instead of
    # probabilistic encoding (mu, log_var) + reparameterization trick
    
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
        "BCE": bce,  # Computed for comparison, though VQVAE doesn't use it
        "VQ_Loss": vq_loss,  # Replaces KL divergence in VQVAE
        "Total_Loss": total_loss,
        "LPIPS": lp,
        "SSIM": ssim_val,
        "GMSD": gmsd_val,
        # Note: KL divergence is not applicable to VQVAE (no probabilistic latent space)
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
    
    # Step 1: Determine the shape that decoder expects by running a forward pass
    dummy_input = torch.randn(1, *model.model_config.input_dim, device=device)
    with torch.no_grad():
        # Run full forward pass to see what shape the decoder receives
        encoder_out = model.encoder(dummy_input)
        embeddings = encoder_out.embedding
        
        # Simulate the forward pass logic exactly as in VQVAE.forward()
        reshape_for_decoding = False
        if len(embeddings.shape) == 2:
            embeddings_reshaped = embeddings.reshape(embeddings.shape[0], 1, 1, -1)
            reshape_for_decoding = True
        else:
            embeddings_reshaped = embeddings
        
        embeddings_permuted = embeddings_reshaped.permute(0, 2, 3, 1)
        
        # Get quantizer output shape
        quantizer_output = model.quantizer(embeddings_permuted, uses_ddp=False)
        quantized_embed = quantizer_output.quantized_vector
        
        if reshape_for_decoding:
            quantized_embed = quantized_embed.reshape(embeddings.shape[0], -1)
        
        decoder_input_shape = quantized_embed.shape[1:]  # Shape without batch dimension
        decoder_input_size = quantized_embed.shape[1]
        
        print(f"Debug: encoder output shape: {embeddings.shape}")
        print(f"Debug: quantized_embed shape before decoder: {quantized_embed.shape}")
        print(f"Debug: decoder_input_size: {decoder_input_size}, embedding_dim: {embedding_dim}")
    
    # Step 2: Sample random quantized embeddings matching the decoder input shape
    # The decoder_input_size tells us how many elements we need
    # Calculate how many spatial locations we need
    num_spatial_locations = decoder_input_size // embedding_dim
    
    if num_spatial_locations == 1:
        # Simple case: one embedding per sample (flattened encoder output)
        random_indices = torch.randint(0, num_embeddings, (num_samples,), device=device)
        z_quant = codebook[random_indices]  # [num_samples, embedding_dim]
        
        # Verify size matches
        if z_quant.shape[1] != decoder_input_size:
            raise ValueError(
                f"Size mismatch: decoder expects {decoder_input_size}, "
                f"but got {z_quant.shape[1]} from single embedding"
            )
    else:
        # Spatial case: need multiple embeddings per sample
        # Sample random indices for each spatial location
        random_indices = torch.randint(0, num_embeddings, 
                                     (num_samples, num_spatial_locations), device=device)
        # Get embeddings: [num_samples, num_spatial_locations, embedding_dim]
        z_spatial = codebook[random_indices]
        
        # Reshape to match what decoder expects
        # The quantizer outputs [B, embedding_dim, H, W] which gets flattened to [B, embedding_dim * H * W]
        # So we need to reshape to [B, embedding_dim, H, W] then flatten
        # Try to infer spatial dimensions (assume square if possible)
        h = w = int(np.sqrt(num_spatial_locations))
        if h * w == num_spatial_locations:
            # Perfect square
            z_spatial = z_spatial.reshape(num_samples, h, w, embedding_dim)
            z_spatial = z_spatial.permute(0, 3, 1, 2)  # [num_samples, embedding_dim, h, w]
            z_quant = z_spatial.reshape(num_samples, -1)  # [num_samples, decoder_input_size]
        else:
            # Not a perfect square, just flatten directly
            z_quant = z_spatial.reshape(num_samples, -1)  # [num_samples, decoder_input_size]
        
        # Verify size matches
        if z_quant.shape[1] != decoder_input_size:
            raise ValueError(
                f"Size mismatch: decoder expects {decoder_input_size}, "
                f"but got {z_quant.shape[1]} from spatial embeddings"
            )
    
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