"""
An image-based algae density analysis tool that uses K-means clustering (an unsupervised machine learning technique) 
to achieve more accurate segmentation compared with simple pixel-by-pixel thresholding methods.

K-means clustering automatically groups similar pixels into different clusters based on their color characteristics. 
By identifying the cluster corresponding to algae, the system can separate algae from water and other objects more effectively 
than using a fixed color threshold.

"""

import argparse  # Command-line arguments
import sys # System functions
from pathlib import Path
import cv2  # Image processing
import numpy as np # Numerical and pixel data
import matplotlib # Visualization
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt # Plotting
from matplotlib.colors import LinearSegmentedColormap # Custom color map
from sklearn.cluster import KMeans # K-means clustering


# ---------------------------------------------------------------------------
# Step 1: Read the image and calculate color features for each pixel
# ---------------------------------------------------------------------------

def load_image(path: Path, max_width: int) -> np.ndarray:

    """Read and resize image if needed; return RGB uint8 array"""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    if w > max_width:
        scale = max_width / w
        rgb = cv2.resize(rgb, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def compute_features(rgb: np.ndarray, denoise: int) -> np.ndarray:
    """
   Extract three pixel features for clustering:
     1. Excess Green → detects green areas.
     2. Saturation → distinguishes water from dense algae.
     3. Value → accounts for brightness and shadows.
Using all three features improves K-means clustering accuracy and stability.
    """
    if denoise > 0:
        k = denoise * 2 + 1
        rgb = cv2.GaussianBlur(rgb, (k, k), 0)

    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    excess_green = 2 * g - r - b  # khoang [-510, 510]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1]  # [0, 255]
    value = hsv[..., 2]       # [0, 255]

    features = np.stack([excess_green, saturation, value], axis=-1)
    return features


# ---------------------------------------------------------------------------
# Step 2: Apply K-means clustering and rank the clusters by their algae likelihood
# ---------------------------------------------------------------------------

def cluster_density(features: np.ndarray, n_clusters: int, seed: int = 42):
    """
        Perform unsupervised clustering in feature space, then rank the clusters
        by their mean Excess Green value to assign density labels from low to high.
        Returns density labels (0..n_clusters-1) for each pixel and the sorted
    """
    h, w, c = features.shape
    flat = features.reshape(-1, c)

    # Standardize features to the same scale before clustering
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-6
    flat_norm = (flat - mean) / std

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(flat_norm)

    # Rank clusters by mean excess green (first feature, unstandardized)
    center_excess_green = []
    for i in range(n_clusters):
        mask = labels == i
        center_excess_green.append(flat[mask, 0].mean() if mask.any() else -1e9)

    order = np.argsort(center_excess_green)  # low -> high
    rank_of_cluster = np.empty(n_clusters, dtype=int)
    for rank, cluster_id in enumerate(order):
        rank_of_cluster[cluster_id] = rank

    density_rank = rank_of_cluster[labels].reshape(h, w)
    return density_rank, order


# ---------------------------------------------------------------------------
# Step 3: Color-code the image based on algae density and export the results
# ---------------------------------------------------------------------------

def build_colormap(name: str):
    if name == "green":
        colors = ["#ffffe5", "#f7fcb9", "#d9f0a3", "#addd8e", "#78c679",
                  "#41ab5d", "#238443", "#006837", "#004529"]
        return LinearSegmentedColormap.from_list("algae_green", colors)

    if name == "thermal":
        colors = ["#08195b", "#256eb1", "#41b6c4", "#a1dab4", "#ffffbf",
                  "#fecc5c", "#fd8d3c", "#e31a1c", "#800026"]
        return LinearSegmentedColormap.from_list("algae_thermal", colors)
    return plt.get_cmap(name)


def render_outputs(rgb, density_rank, n_clusters, colormap_name, opacity,
                    output_path: Path, report_path: Path):
    cmap = build_colormap(colormap_name)
    norm_rank = density_rank.astype(np.float32) / max(1, n_clusters - 1)

    heat_rgba = cmap(norm_rank)          # (H, W, 4), value 0..1
    heat_rgb = (heat_rgba[..., :3] * 255).astype(np.float32)

    blended = (1 - opacity) * rgb.astype(np.float32) + opacity * heat_rgb
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Result image (only the density map blended with the original image)
    cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    # Visual report: original image | density map | legend + area statistics
    counts = np.array([(density_rank == i).sum() for i in range(n_clusters)])
    percentages = 100.0 * counts / counts.sum()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1, 0.6]})

    axes[0].imshow(rgb)
    axes[0].set_title("Anh goc")
    axes[0].axis("off")

    axes[1].imshow(blended)
    axes[1].set_title("Ban do mat do tao")
    axes[1].axis("off")

    axes[2].axis("off")
    labels = [f"Cum {i + 1} (nhat)" if i == 0 else
              f"Cum {i + 1} (dam nhat)" if i == n_clusters - 1 else
              f"Cum {i + 1}" for i in range(n_clusters)]
    bar_colors = [cmap(i / max(1, n_clusters - 1)) for i in range(n_clusters)]
    y_pos = np.arange(n_clusters)
    axes[2].barh(y_pos, percentages, color=bar_colors)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(labels, fontsize=9)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Ty le dien tich (%)")
    axes[2].set_title("Phan bo mat do")
    for i, v in enumerate(percentages):
        axes[2].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(report_path, dpi=150)
    plt.close(fig)

    return percentages


# ---------------------------------------------------------------------------
# Main 
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze algae density from an image using K-means."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input image"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("density_map.png"),
        help="Path to save the density map image (default: density_map.png)"
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path("analysis_report.png"),
        help="Path to save the visual analysis report (default: analysis_report.png)"
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of density levels to classify (default: 5)"
    )

    parser.add_argument(
        "--colormap",
        choices=["green", "thermal"],
        default="green",
        help="Color map for visualization (default: green)"
    )

    parser.add_argument(
        "--opacity",
        type=float,
        default=0.75,
        help="Opacity of the density map over the original image, 0..1 (default: 0.75)"
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=900,
        help="Maximum processing width; larger images will be resized (default: 900)"
    )

    parser.add_argument(
        "--denoise",
        type=int,
        default=1,
        help="Gaussian blur level before clustering, 0 = disabled (default: 1)"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: file not found {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading image: {args.input}")
    rgb = load_image(args.input, args.max_width)

    print("Computing color features (excess-green, saturation, brightness)...")
    features = compute_features(rgb, args.denoise)

    print(f"Applying K-means clustering with {args.clusters} density levels...")
    density_rank, _ = cluster_density(features, args.clusters)

    print("Coloring and exporting results...")
    percentages = render_outputs(
        rgb, density_rank, args.clusters, args.colormap, args.opacity,
        args.output, args.report,
    )

    print("\nCompleted. Area distribution by density level (low -> high):")
    for i, pct in enumerate(percentages):
        print(f"  Level {i + 1}: {pct:.1f}%")

    print(f"\nDensity map: {args.output}")
    print(f"Visual analysis report: {args.report}")


if __name__ == "__main__":
    main()