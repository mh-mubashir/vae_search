import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def _get_metrics():
    """Return the numerical values used in slide 11 as structured arrays."""
    models = ["β=0.5", "β=1", "β=4", "VQ-VAE", "VAEGAN"]

    # Reconstruction metrics table (slide 11, top table)
    # Columns: BCE, MSE, KL, ELBO, LPIPS, SSIM, GMSD
    recon = {
        "β=0.5": dict(BCE=23.214, MSE=0.015, KL=45.934, ELBO=46.181, LPIPS=0.289, SSIM=0.643, GMSD=0.163),
        "β=1": dict(BCE=32.491, MSE=0.021, KL=14.912, ELBO=47.402, LPIPS=0.318, SSIM=0.565, GMSD=0.190),
        "β=4": dict(BCE=50.755, MSE=0.033, KL=1.090, ELBO=55.115, LPIPS=0.380, SSIM=0.477, GMSD=0.224),
        "VQ-VAE": dict(BCE=np.nan, MSE=0.007, KL=np.nan, ELBO=np.nan, LPIPS=0.260, SSIM=0.687, GMSD=0.155),
        "VAEGAN": dict(BCE=0.508, MSE=0.009, KL=162.584, ELBO=162.592, LPIPS=0.301, SSIM=0.661, GMSD=0.160),
    }

    # Generation metrics table (slide 11, bottom table)
    gen = {
        "β=0.5": dict(BCE=8.764, MSE=0.006, KL=39.163, ELBO=28.345, LPIPS=0.093, SSIM=0.870, GMSD=0.109),
        "β=1": dict(BCE=9.131, MSE=0.006, KL=12.443, ELBO=21.574, LPIPS=0.090, SSIM=0.835, GMSD=0.121),
        "β=4": dict(BCE=12.050, MSE=0.008, KL=0.945, ELBO=15.831, LPIPS=0.073, SSIM=0.841, GMSD=0.124),
        "VQ-VAE": dict(BCE=np.nan, MSE=0.009, KL=np.nan, ELBO=np.nan, LPIPS=0.154, SSIM=0.801, GMSD=0.113),
        "VAEGAN": dict(BCE=0.584, MSE=0.001, KL=165.405, ELBO=165.406, LPIPS=0.033, SSIM=0.970, GMSD=0.031),
    }

    return models, recon, gen


def make_radar_chart(output_path: Path):
    """Visual 1: Radar chart – reconstruction metrics (MSE, LPIPS, SSIM, GMSD)."""
    models, recon, _ = _get_metrics()
    metric_names = ["MSE", "LPIPS", "SSIM", "GMSD"]

    raw = np.array([[recon[m][k] for k in metric_names] for m in models], dtype=float)

    # For MSE, LPIPS, GMSD lower is better; for SSIM higher is better.
    better_is_lower = np.array([True, True, False, True])
    mins = np.nanmin(raw, axis=0)
    maxs = np.nanmax(raw, axis=0)

    scores = np.zeros_like(raw)
    for j in range(len(metric_names)):
        if maxs[j] == mins[j]:
            scores[:, j] = 0.5
        else:
            if better_is_lower[j]:
                scores[:, j] = (maxs[j] - raw[:, j]) / (maxs[j] - mins[j])
            else:
                scores[:, j] = (raw[:, j] - mins[j]) / (maxs[j] - mins[j])

    num_vars = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(5, 5))

    for i, model in enumerate(models):
        values = scores[i].tolist()
        values += values[:1]
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_names)
    ax.set_ylim(0, 1)
    ax.set_title("Reconstruction metric landscape", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_kl_collapse_bar(output_path: Path):
    """Visual 2: Bar chart – KL collapse across β values (reconstruction metrics)."""
    _, recon, _ = _get_metrics()
    betas = np.array([0.5, 1.0, 4.0])
    kl_vals = np.array([recon["β=0.5"]["KL"], recon["β=1"]["KL"], recon["β=4"]["KL"]])

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(["0.5", "1", "4"], kl_vals, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("KL divergence")
    ax.set_title("KL collapse with increasing $\\beta$")

    for i, v in enumerate(kl_vals):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_quality_clusters_strip(output_path: Path):
    """Visual 3: Side-by-side 'quality clusters' schematic."""
    labels = ["β-VAEs", "VQ-VAE", "VAEGAN"]
    descriptions = [
        "Trade-off: fidelity ↓,\n        disentanglement ↑",
        "Discrete latent\nsweet spot",
        "Perceptual realism\npeak",
    ]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis("off")

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    x_positions = [0.1, 0.4, 0.7]

    for x, label, desc, color in zip(x_positions, labels, descriptions, colors):
        rect = patches.FancyBboxPatch(
            (x, 0.15),
            0.22,
            0.7,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor=color,
            facecolor=color + "20" if isinstance(color, str) else color,
        )
        ax.add_patch(rect)
        ax.text(x + 0.11, 0.7, label, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 0.11, 0.4, desc, ha="center", va="center", fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Quality clusters across models", pad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_beta_tradeoff_curve(output_path: Path):
    """Slide 12, Visual A: trade-off curve over β (KL vs reconstruction/perceptual error)."""
    _, recon, _ = _get_metrics()
    betas = np.array([0.5, 1.0, 4.0])
    kl = np.array([recon["β=0.5"]["KL"], recon["β=1"]["KL"], recon["β=4"]["KL"]])
    mse = np.array([recon["β=0.5"]["MSE"], recon["β=1"]["MSE"], recon["β=4"]["MSE"]])
    lpips = np.array([recon["β=0.5"]["LPIPS"], recon["β=1"]["LPIPS"], recon["β=4"]["LPIPS"]])

    # Normalize to [0, 1] for a clean visual crossing.
    def norm(x):
        return (x - x.min()) / (x.max() - x.min())

    kl_n = norm(kl[::-1])[::-1]  # visually steep downward
    recon_err = norm(mse + lpips)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(betas, kl_n, "-o", color="#4C72B0", label="KL (capacity ↓)")
    ax.plot(betas, recon_err, "-o", color="#C44E52", label="Reconstruction / LPIPS (error ↑)")

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Normalized value")
    ax.set_xticks(betas)
    ax.set_title(r"Trade-off between KL capacity and reconstruction quality")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_beta_disentanglement_meter(output_path: Path):
    """Slide 12, Visual B: 'Disentanglement vs. Fidelity' meter."""
    betas = ["0.5", "1", "4"]
    positions = [0.2, 0.5, 0.8]  # conceptual positions along Fidelity ↔ Disentanglement
    colors = ["#4C72B0", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.axhline(0, color="lightgray", linewidth=8, zorder=0)

    for pos, beta, color in zip(positions, betas, colors):
        ax.scatter(pos, 0, s=200, color=color, edgecolor="black", zorder=2)
        ax.text(pos, 0.25, rf"$\beta={beta}$", ha="center", va="bottom", fontsize=9)

    ax.text(0.02, -0.35, "Fidelity ↑", ha="left", va="center", fontsize=9)
    ax.text(0.98, -0.35, "Disentanglement ↑↑", ha="right", va="center", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    ax.set_title(r"Disentanglement vs. fidelity along $\beta$", pad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_vqvae_recon_bars(output_path: Path):
    """Slide 13, Visual A: reconstruction quality bar pairs (MSE and SSIM)."""
    models, recon, _ = _get_metrics()
    mse_vals = np.array([recon[m]["MSE"] for m in models])
    ssim_vals = np.array([recon[m]["SSIM"] for m in models])

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(6, 3))

    # Left axis: MSE (lower is better)
    bars1 = ax1.bar(x - width / 2, mse_vals, width, label="MSE", color="#4C72B0")
    ax1.set_ylabel("MSE (↓)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)

    # Right axis: SSIM (higher is better)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, ssim_vals, width, label="SSIM", color="#55A868", alpha=0.8)
    ax2.set_ylabel("SSIM (↑)")

    ax1.set_title("Reconstruction MSE and SSIM across models")

    # Combined legend
    handles = [bars1, bars2]
    labels = ["MSE", "SSIM"]
    ax1.legend(handles, labels, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_codebook_usage_diagram(output_path: Path):
    """Slide 13, Visual B: simple mosaic for codebook usage (reuse vs flexibility)."""
    rng = np.random.default_rng(0)

    # Left: high reuse (few unique codes)
    left = rng.integers(0, 4, size=(8, 8))
    # Right: more flexible (more unique codes)
    right = rng.integers(0, 12, size=(8, 8))

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    im0 = axes[0].imshow(left, cmap="tab20")
    axes[0].set_title("High codebook reuse")
    axes[0].axis("off")

    im1 = axes[1].imshow(right, cmap="tab20")
    axes[1].set_title("Flexible representation")
    axes[1].axis("off")

    fig.suptitle("VQ-VAE codebook usage schematic", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_model_comparison_strip(output_path: Path):
    """Slide 14 / 15: three-model visual strip (β-VAE, VQ-VAE, VAEGAN) using generation metrics.

    Left panel: generation MSE (↓) and SSIM (↑) vs β.
    Middle panel: VQ-VAE generation MSE (↓) and SSIM (↑).
    Right panel: VAEGAN generation SSIM (↑) and LPIPS (↓).
    All values are taken from the *generation* table (slide 11, bottom).
    """
    _, _, gen = _get_metrics()

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    titles = [r"$\beta$-VAE", "VQ-VAE", "VAEGAN"]
    subtitles = [
        "Latent → interpretable\nImages → blurrier",
        "Discrete latent\nGood balance",
        "Best realism\nLatents harder to interpret",
    ]

    # Simple schematic lines for each model using generation metrics
    beta_mse = [gen["β=0.5"]["MSE"], gen["β=1"]["MSE"], gen["β=4"]["MSE"]]
    beta_ssim = [gen["β=0.5"]["SSIM"], gen["β=1"]["SSIM"], gen["β=4"]["SSIM"]]
    vq_mse, vq_ssim = gen["VQ-VAE"]["MSE"], gen["VQ-VAE"]["SSIM"]
    gan_mse, gan_ssim = gen["VAEGAN"]["MSE"], gen["VAEGAN"]["SSIM"]

    # Panel 1: β-VAE – generation MSE and SSIM vs β
    ax = axes[0]
    x = [0.5, 1, 4]
    line_mse, = ax.plot(x, beta_mse, "-o", color="#4C72B0")
    ax2 = ax.twinx()
    line_ssim, = ax2.plot(x, beta_ssim, "-s", color="#55A868")
    ax.set_xticks(x)
    ax.set_xlabel(r"$\beta$", fontsize=9)
    ax.set_ylabel("Generation MSE (↓)", fontsize=9)
    ax2.set_ylabel("Generation SSIM (↑)", fontsize=9)
    ax.set_title(titles[0], fontsize=12)
    ax.tick_params(axis="both", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    ax.text(0.5, -0.5, subtitles[0], ha="left", va="top", transform=ax.transAxes, fontsize=7)

    # Panel 2: VQ-VAE – single-model generation MSE and SSIM
    ax = axes[1]
    bars_vq_mse = ax.bar([0], [vq_mse], width=0.5, color="#4C72B0")
    ax2 = ax.twinx()
    bars_vq_ssim = ax2.bar([0.6], [vq_ssim], width=0.5, color="#55A868", alpha=0.8)
    ax.set_xticks([0, 0.6])
    ax.set_xticklabels(["MSE", "SSIM"], fontsize=8)
    ax.set_ylabel("Generation MSE (↓)", fontsize=9)
    ax2.set_ylabel("Generation SSIM (↑)", fontsize=9)
    ax.set_title(titles[1], fontsize=12)
    ax.tick_params(axis="y", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    ax.text(0.5, -0.5, subtitles[1], ha="center", va="top", transform=ax.transAxes, fontsize=7)

    # Panel 3: VAEGAN – highlight generation SSIM and LPIPS
    ax = axes[2]
    bars_gan_ssim = ax.bar([0], [gen["VAEGAN"]["SSIM"]], color="#55A868")
    ax2 = ax.twinx()
    bars_gan_lpips = ax2.bar([0.6], [gen["VAEGAN"]["LPIPS"]], color="#C44E52", alpha=0.8)
    ax.set_xticks([0, 0.6])
    ax.set_xticklabels(["SSIM", "LPIPS"], fontsize=8)
    ax.set_ylabel("Generation SSIM (↑)", fontsize=9)
    ax2.set_ylabel("Generation LPIPS (↓)", fontsize=9)
    ax.set_title(titles[2], fontsize=12)
    ax.tick_params(axis="y", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    ax.text(0.5, -0.5, subtitles[2], ha="center", va="top", transform=ax.transAxes, fontsize=7)

    # Shared, compact legend: color ↔ metric family
    handles = [line_mse, bars_vq_ssim, bars_gan_lpips]
    labels = ["MSE (↓, blue)", "SSIM (↑, green)", "LPIPS (↓, red)"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Model comparison strip: generation reconstruction vs perceptual quality",
        fontsize=12,
        y=0.95,
    )
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.9])
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_accuracy_disentanglement_realism_triangle(output_path: Path):
    """Slide 14: Accuracy–Disentanglement–Realism triangle."""
    fig, ax = plt.subplots(figsize=(4, 3.5))

    # Coordinates of an equilateral triangle
    pts = np.array(
        [
            [0.0, 0.0],  # left
            [1.0, 0.0],  # right
            [0.5, math.sqrt(3) / 2.0],  # top
        ]
    )
    tri = patches.Polygon(pts, closed=True, fill=False, linewidth=2)
    ax.add_patch(tri)

    # Vertex labels
    ax.text(pts[2, 0], pts[2, 1] + 0.05, "Interpretability\n(β-VAE)", ha="center", va="bottom", fontsize=9)
    ax.text(pts[0, 0] - 0.05, pts[0, 1] - 0.05, "Reconstruction\nfidelity (VQ-VAE)", ha="right", va="top", fontsize=9)
    ax.text(pts[1, 0] + 0.05, pts[1, 1] - 0.05, "Realistic\nsamples (VAEGAN)", ha="left", va="top", fontsize=9)

    # Place approximate model points near their respective corners
    ax.scatter(0.45, 0.65, color="#4C72B0", label=r"$\beta$-VAE")
    ax.scatter(0.3, 0.15, color="#55A868", label="VQ-VAE")
    ax.scatter(0.75, 0.15, color="#C44E52", label="VAEGAN")

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.1)
    ax.axis("off")
    ax.set_title("Accuracy–disentanglement–realism triangle", pad=12)
    ax.legend(loc="lower center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_all_figures(output_dir: Path | None = None):
    """Generate all figures needed for slides 11B–14."""
    if output_dir is None:
        output_dir = BASE_DIR / "Figures"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    make_radar_chart(output_dir / "metric_radar_recon.png")
    make_kl_collapse_bar(output_dir / "kl_collapse_bar.png")
    make_quality_clusters_strip(output_dir / "quality_clusters_strip.png")

    make_beta_tradeoff_curve(output_dir / "beta_tradeoff_curve.png")
    make_beta_disentanglement_meter(output_dir / "beta_disentanglement_meter.png")

    make_vqvae_recon_bars(output_dir / "vqvae_recon_ssim_bars.png")
    make_codebook_usage_diagram(output_dir / "vqvae_codebook_usage.png")

    make_model_comparison_strip(output_dir / "model_comparison_strip.png")
    make_accuracy_disentanglement_realism_triangle(
        output_dir / "accuracy_disentanglement_realism_triangle.png"
    )


if __name__ == "__main__":
    make_all_figures()


