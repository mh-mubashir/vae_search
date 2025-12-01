import csv
import os

from PIL import Image


def main():
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(project_root, "graph_generation", "vqvae", "vqvae.csv")
    figures_dir = os.path.join(project_root, "graph_generation", "vqvae", "figures")
    output_path = os.path.join(figures_dir, "vqvae_recon_gen_grid.png")

    # Read CSV and collect first 8 entries (orig, recon, gen)
    originals = []
    recons = []
    gens = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Use up to 10 images (img_0 ... img_9)
            if i >= 10:
                break
            # CSV paths are like "media/images/xxx.png" – map to presentation/figures/xxx.png
            orig_name = os.path.basename(row["truthfilepath"].strip('"'))
            recon_name = os.path.basename(row["reconstructionfilepath"].strip('"'))
            gen_name = os.path.basename(row["normal_generationfilepath"].strip('"'))

            originals.append(os.path.join(figures_dir, orig_name))
            recons.append(os.path.join(figures_dir, recon_name))
            gens.append(os.path.join(figures_dir, gen_name))

    if len(originals) == 0:
        raise RuntimeError("No rows read from vqvae.csv – check the CSV format.")

    # Load one image to determine size
    sample_img = Image.open(originals[0]).convert("RGB")
    w, h = sample_img.size

    cols = len(originals)
    rows = 3  # originals, reconstructions, generations
    grid_w = cols * w
    grid_h = rows * h

    # Create base grid without labels (white background like paper figures)
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    # Helper to paste a row of images
    def paste_row(img_paths, row_idx):
        y_offset = row_idx * h
        for col_idx, path in enumerate(img_paths):
            if col_idx >= cols:
                break
            img = Image.open(path).convert("RGB")
            # ensure consistent size
            if img.size != (w, h):
                img = img.resize((w, h), Image.BILINEAR)
            x_offset = col_idx * w
            grid.paste(img, (x_offset, y_offset))

    paste_row(originals, 0)
    paste_row(recons, 1)
    paste_row(gens, 2)

    # Add left-side labels: "Orig", "Recon", "Gen"
    from PIL import ImageDraw, ImageFont

    # Make label column wide enough for full words like "Reconstruction"
    label_width = int(3 * h)
    canvas_w = grid_w + label_width
    canvas_h = grid_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    # Paste the grid shifted to the right
    canvas.paste(grid, (label_width, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=int(0.3 * h))
    except Exception:
        font = ImageFont.load_default()

    labels = ["Original", "Reconstruction", "Generated"]
    for row_idx, text in enumerate(labels):
        # Compute text bounding box to determine width/height (Pillow >=10 removed textsize)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            # Fallback for very old Pillow versions
            text_w, text_h = font.getsize(text)

        y_center = row_idx * h + h // 2
        x = max(5, (label_width - text_w) // 2)
        y = int(y_center - text_h / 2)
        draw.text((x, y), text, fill=(0, 0, 0), font=font)

    canvas.save(output_path)
    print(f"Saved VQ-VAE recon/gen grid with labels to: {output_path}")


if __name__ == "__main__":
    main()


