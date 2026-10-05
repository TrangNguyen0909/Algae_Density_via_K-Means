"""
algae_density.py - Image-based algae density analysis using K-means (upgraded version)
======================================================================================
 
Pipeline:
  1. Read the image (Unicode paths supported), resize, denoise.
  2. EXCLUDE unreliable regions from the analysis:
       - Glare / specular reflection / white foam / clouds (bright + low saturation)
       - Deep shadows
       - Strongly saturated red / orange / purple / pink objects (boats, signs, clothing...)
       - Areas outside the ROI (user-defined via mask or polygon)
       - Manually excluded areas (user-supplied mask)
  3. Compute illumination-robust color features (chromaticity ExG, Lab a*, S, V).
  4. Run K-means ONLY on valid pixels (fit on a subsample for speed, predict all).
  5. Rank clusters by ABSOLUTE greenness; clusters below the threshold become
     "no algae / water". (K-means always forces k clusters, so without an absolute
     threshold an algae-free image would still be reported as having "dense algae".)
  6. Smooth labels, remove tiny blobs, export density map + report + statistics (JSON/CSV).
 
"""
 
from __future__ import annotations
 
import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
 
import cv2
import numpy as np
import matplotlib
 
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
 
log = logging.getLogger("algae")
 
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CMAPS = {"green": "YlGn", "thermal": "turbo"}
 
# Feature weights: [ExG, -a*, Saturation, Value]. Greenness features are prioritized.
FEATURE_WEIGHTS = np.array([1.5, 1.0, 0.6, 0.6], dtype=np.float32)
 
# Color and display name of each excluded-region type
EXCL_STYLE = {
    "glare": ((255, 220, 0), "Glare / reflection / white foam"),
    "shadow": ((60, 90, 255), "Deep shadow"),
    "object": ((255, 60, 60), "Red / purple objects"),
    "outside_roi": ((150, 150, 150), "Outside ROI"),
    "user_exclude": ((255, 0, 255), "Manually excluded"),
}
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
@dataclass
class Config:
    clusters: int = 5
    max_width: int = 900
    denoise: int = 1
    algae_threshold: float = 0.04   # Minimum chromaticity ExG for a cluster to count as algae
    sample_size: int = 200_000
    seed: int = 42
    smooth: int = 5                 # Median filter size (0 = off)
    min_blob: int = 30              # Minimum algae blob area (pixels)
    min_valid_frac: float = 0.15    # Minimum fraction of valid pixels required
    filter_glare: bool = True
    filter_shadow: bool = True
    filter_objects: bool = True
    glare_v: int = 235
    glare_s: int = 40
    shadow_v: int = 35
    roi_mask: Optional[Path] = None
    roi_polygon: Optional[str] = None
    exclude_mask: Optional[Path] = None
    opacity: float = 0.7
    colormap: str = "green"
 
 
# ---------------------------------------------------------------------------
# Step 1: Image I/O and preprocessing
# ---------------------------------------------------------------------------
 
def read_bgr(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image safely with Unicode paths (cv2.imread often fails on Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags) if data.size else None
    if img is None:
        raise ValueError(f"Could not read/decode image: {path}")
    return img
 
 
def write_image(path: Path, bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix or ".png", bgr)
    if not ok:
        raise IOError(f"Could not write image: {path}")
    buf.tofile(str(path))
 
 
def resize_rgb(rgb: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale an RGB image if wider than max_width (<= 0 : keep original size)."""
    h, w = rgb.shape[:2]
    if max_width > 0 and w > max_width:
        scale = max_width / w
        rgb = cv2.resize(rgb, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return rgb
 
 
def load_image(path: Path, max_width: int) -> np.ndarray:
    """Read an image, convert to RGB uint8 and downscale if too wide."""
    return resize_rgb(cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2RGB), max_width)
 
 
def denoise_image(rgb: np.ndarray, level: int) -> np.ndarray:
    if level <= 0:
        return rgb
    k = level * 2 + 1
    return cv2.GaussianBlur(rgb, (k, k), 0)
 
 
# ---------------------------------------------------------------------------
# Step 2: Build exclusion masks
# ---------------------------------------------------------------------------
 
def file_to_mask(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    """Read a mask image (white = True) and resize it to the working image size."""
    gray = read_bgr(path, cv2.IMREAD_GRAYSCALE)
    h, w = shape
    gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)
    return gray > 127
 
 
def polygon_to_mask(spec: str, shape: Tuple[int, int]) -> np.ndarray:
    """Polygon in normalized 0..1 coordinates: 'x1,y1;x2,y2;x3,y3;...' (origin at top-left)."""
    h, w = shape
    try:
        pts = [tuple(float(v) for v in p.split(",")) for p in spec.split(";") if p.strip()]
        if any(len(p) != 2 for p in pts):
            raise ValueError
    except ValueError:
        raise ValueError("--roi-polygon has an invalid format; expected 'x1,y1;x2,y2;x3,y3' (0..1)")
    if len(pts) < 3:
        raise ValueError("--roi-polygon needs at least 3 points")
    if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in pts):
        raise ValueError("--roi-polygon: coordinates must be within 0..1")
    arr = np.array([[x * (w - 1), y * (h - 1)] for x, y in pts], dtype=np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [arr], 255)
    return m > 0
 
 
def build_exclusion(rgb: np.ndarray, cfg: Config) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Return (per-type masks, combined exclusion mask). True = excluded."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[..., 0].astype(np.int16)  # OpenCV: 0..179
    S = hsv[..., 1].astype(np.int16)
    V = hsv[..., 2].astype(np.int16)
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    masks: Dict[str, np.ndarray] = {}
 
    if cfg.filter_glare:
        # Very bright + almost no color => sun glare, foam, clouds, sky.
        g = ((V >= cfg.glare_v) & (S <= cfg.glare_s)).astype(np.uint8)
        masks["glare"] = cv2.dilate(g, k5) > 0  # dilate to cover the halo around glare spots
 
    if cfg.filter_shadow:
        s = (V <= cfg.shadow_v).astype(np.uint8)
        masks["shadow"] = cv2.morphologyEx(s, cv2.MORPH_OPEN, k3) > 0  # drop isolated noise pixels
 
    if cfg.filter_objects:
        # Strongly saturated red / dark orange / purple / pink: rarely algae (boats, buoys, signs...).
        o = (((H <= 8) | (H >= 140)) & (S >= 90) & (V >= 50)).astype(np.uint8)
        o = cv2.morphologyEx(o, cv2.MORPH_OPEN, k3)
        masks["object"] = cv2.morphologyEx(o, cv2.MORPH_CLOSE, k5) > 0
 
    roi = np.ones((h, w), bool)
    if cfg.roi_mask is not None:
        roi &= file_to_mask(cfg.roi_mask, (h, w))
    if cfg.roi_polygon:
        roi &= polygon_to_mask(cfg.roi_polygon, (h, w))
    if cfg.roi_mask is not None or cfg.roi_polygon:
        masks["outside_roi"] = ~roi
 
    if cfg.exclude_mask is not None:
        masks["user_exclude"] = file_to_mask(cfg.exclude_mask, (h, w))
 
    excluded = np.zeros((h, w), bool)
    for m in masks.values():
        excluded |= m
    return masks, excluded
 
 
# ---------------------------------------------------------------------------
# Step 3: Color features
# ---------------------------------------------------------------------------
 
def compute_features(rgb: np.ndarray) -> np.ndarray:
    """
    Four features per pixel:
      0. Chromaticity ExG = 2g' - r' - b' (r'=R/(R+G+B), ...): greenness, largely
         independent of illumination intensity.
      1. -a* (Lab): more positive = greener.
      2. Saturation (0..1): separates pale water from deeply colored algae.
      3. Value (0..1): brightness.
    """
    rgb_f = rgb.astype(np.float32)
    total = rgb_f.sum(axis=-1) + 1e-6
    rc, gc, bc = (rgb_f[..., i] / total for i in range(3))
    exg = 2 * gc - rc - bc  # range [-1, 2]
 
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    neg_a = -(lab[..., 1] - 128.0) / 128.0
 
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    return np.stack([exg, neg_a, sat, val], axis=-1)
 
 
# ---------------------------------------------------------------------------
# Step 4: K-means on valid pixels and density level assignment
# ---------------------------------------------------------------------------
 
def cluster_valid_pixels(features: np.ndarray, valid: np.ndarray, cfg: Config):
    """Return (cluster label of each valid pixel, mean ExG per cluster, actual k)."""
    X = features[valid]
    n = X.shape[0]
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    Xn = ((X - mu) / sd) * FEATURE_WEIGHTS
 
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(n, size=min(n, cfg.sample_size), replace=False)
 
    k = int(max(1, min(cfg.clusters, len(idx) // 50)))
    km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed)
    km.fit(Xn[idx])
    labels = km.predict(Xn)  # label ALL valid pixels
 
    exg_means = np.array(
        [X[labels == i, 0].mean() if np.any(labels == i) else -1e9 for i in range(k)]
    )
    return labels, exg_means, k
 
 
def assign_levels(exg_means: np.ndarray, threshold: float) -> Tuple[np.ndarray, int]:
    """
    Level 0 = no algae / water (ExG below threshold). Levels 1..m = increasing density by ExG.
    Returns (level of each cluster, number of algae levels m).
    """
    level_of_cluster = np.zeros(len(exg_means), dtype=np.uint8)
    lvl = 0
    for c in np.argsort(exg_means):
        if exg_means[c] >= threshold:
            lvl += 1
            level_of_cluster[c] = lvl
    return level_of_cluster, lvl
 
 
def smooth_levels(levels: np.ndarray, valid: np.ndarray, ksize: int) -> np.ndarray:
    if ksize < 3:
        return levels
    if ksize % 2 == 0:
        ksize += 1
    out = cv2.medianBlur(levels, ksize)
    out[~valid] = 0
    return out
 
 
def remove_small_blobs(levels: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return levels
    mask = (levels > 0).astype(np.uint8)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
    if small.size:
        levels = levels.copy()
        levels[np.isin(lab, small)] = 0
    return levels
 
 
# ---------------------------------------------------------------------------
# Step 5: Statistics
# ---------------------------------------------------------------------------
 
def classify_coverage(cover_pct: float, n_levels: int) -> str:
    """Empirical classification by % algae cover (calibrate against field data)."""
    if n_levels == 0:
        return "No algae detected"
    if cover_pct < 1:
        return "Negligible"
    if cover_pct < 10:
        return "Low"
    if cover_pct < 30:
        return "Moderate"
    if cover_pct < 60:
        return "High"
    return "Very high"
 
 
def level_labels(n_levels: int) -> List[str]:
    labels = ["No algae / water"]
    for i in range(1, n_levels + 1):
        if n_levels == 1:
            labels.append("Algae present")
        elif i == 1:
            labels.append(f"Level {i} (sparsest)")
        elif i == n_levels:
            labels.append(f"Level {i} (densest)")
        else:
            labels.append(f"Level {i}")
    return labels
 
 
def compute_stats(levels, valid, features, n_levels, masks, path, shape, warnings) -> dict:
    valid_px = int(valid.sum())
    exg = features[..., 0]
    counts = np.array([int(((levels == i) & valid).sum()) for i in range(n_levels + 1)])
    pct = 100.0 * counts / max(valid_px, 1)
    cover = float(100.0 * (valid & (levels > 0)).sum() / max(valid_px, 1))
    density_index = (
        float(levels[valid].sum() / (n_levels * valid_px)) if n_levels > 0 else 0.0
    )
    algae_px = valid & (levels > 0)
    level_exg = [
        float(exg[valid & (levels == i)].mean()) if counts[i] > 0 else None
        for i in range(n_levels + 1)
    ]
    return {
        "file": str(path),
        "image_size": f"{shape[1]}x{shape[0]}",
        "valid_area_pct": round(100.0 * valid_px / valid.size, 2),
        "excluded_pct_by_type": {k: round(100.0 * float(m.mean()), 2) for k, m in masks.items()},
        "n_algae_levels": int(n_levels),
        "algae_cover_pct": round(cover, 2),
        "density_index": round(density_index, 4),   # 0..1, mean level / max level
        "mean_exg_valid": round(float(exg[valid].mean()), 4),  # comparable across images
        "mean_exg_algae": round(float(exg[algae_px].mean()), 4) if algae_px.any() else None,
        "category": classify_coverage(cover, n_levels),
        "level_pct": [round(float(p), 2) for p in pct],
        "level_mean_exg": [None if v is None else round(v, 4) for v in level_exg],
        "warnings": warnings,
    }
 
 
# ---------------------------------------------------------------------------
# Step 6: Visualization and export
# ---------------------------------------------------------------------------
 
def level_palette(n_levels: int, cmap_name: str) -> np.ndarray:
    cmap = plt.get_cmap(CMAPS[cmap_name])
    pos = [0.8] if n_levels == 1 else np.linspace(0.35, 1.0, max(n_levels, 1))
    cols = np.zeros((n_levels + 1, 3), np.float32)
    for i in range(1, n_levels + 1):
        cols[i] = np.array(cmap(float(pos[i - 1]))[:3]) * 255
    return cols
 
 
def make_overlay(rgb, levels, excluded, palette, opacity) -> np.ndarray:
    out = rgb.astype(np.float32)
    m = (levels > 0) & ~excluded
    out[m] = (1 - opacity) * out[m] + opacity * palette[levels[m]]
    out[excluded] = 0.45 * out[excluded] + 0.55 * np.array([110, 110, 110], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
 
 
def make_exclusion_view(rgb, masks) -> np.ndarray:
    out = (rgb.astype(np.float32) * 0.6)
    for key, (color, _) in EXCL_STYLE.items():
        if key in masks:
            m = masks[key]
            out[m] = 0.35 * out[m] + 0.65 * np.array(color, np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
 
 
def render_report(rgb, masks, overlay, stats, palette, report_path) -> None:
    """report_path may be a file path or a file-like object (e.g. io.BytesIO)."""
    n_levels = stats["n_algae_levels"]
    labels = level_labels(n_levels)
    pct = stats["level_pct"]
    bar_colors = [np.array([158, 202, 225]) / 255.0] + [palette[i] / 255.0 for i in range(1, n_levels + 1)]
 
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("Original image")
 
    ax[0, 1].imshow(make_exclusion_view(rgb, masks))
    ax[0, 1].set_title(f"Excluded regions ({stats['valid_area_pct']:.1f}% of area remains)")
    handles = [Patch(color=np.array(c) / 255.0, label=n) for k, (c, n) in EXCL_STYLE.items() if k in masks]
    if handles:
        ax[0, 1].legend(handles=handles, loc="lower center", fontsize=8, ncol=2, framealpha=0.85)
 
    ax[1, 0].imshow(overlay)
    ax[1, 0].set_title("Algae density map")
 
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.axis("off")
 
    y = np.arange(len(pct))
    ax[1, 1].barh(y, pct, color=bar_colors)
    ax[1, 1].set_yticks(y)
    ax[1, 1].set_yticklabels(labels, fontsize=9)
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_xlabel("Share of valid area (%)")
    ax[1, 1].set_title("Density distribution")
    for i, v in enumerate(pct):
        ax[1, 1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax[1, 1].set_xlim(0, max(pct) * 1.18 + 1)
 
    fig.suptitle(
        f"Algae cover: {stats['algae_cover_pct']:.1f}%  |  Density index: {stats['density_index']:.2f}  |  "
        f"Rating: {stats['category']}",
        fontsize=13, fontweight="bold",
    )
    if stats["warnings"]:
        fig.text(0.5, 0.01, "Warning: " + " | ".join(stats["warnings"]), ha="center",
                 fontsize=8, color="#b00020", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(report_path, dpi=150)
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Single-image pipeline
# ---------------------------------------------------------------------------
 
def analyze_rgb(rgb: np.ndarray, cfg: Config, name: str = "image") -> dict:
    """
    Core analysis on a (resized) RGB uint8 array. Shared by the CLI and the web UI.
    Returns a dict: rgb, stats, levels, masks, excluded, overlay, palette.
    """
    warnings: List[str] = []
    blur = denoise_image(rgb, cfg.denoise)
 
    masks, excluded = build_exclusion(blur, cfg)
    valid = ~excluded
    valid_frac = float(valid.mean())
    if valid_frac < cfg.min_valid_frac:
        raise ValueError(
            f"Only {valid_frac * 100:.1f}% of pixels remain valid (< {cfg.min_valid_frac * 100:.0f}%). "
            "The image is too glary/dark/obstructed, or the ROI is too small."
        )
    if "glare" in masks and masks["glare"].mean() > 0.4:
        warnings.append("More than 40% of the image is glare/reflection; results are less reliable")
 
    features = compute_features(blur)
    if features[..., 0][valid].std() < 0.01:
        warnings.append("Image colors are nearly uniform; density levels may only reflect noise")
 
    labels, exg_means, k = cluster_valid_pixels(features, valid, cfg)
    if k < cfg.clusters:
        warnings.append(f"Only {k}/{cfg.clusters} clusters could be formed due to limited data")
 
    level_of_cluster, n_levels = assign_levels(exg_means, cfg.algae_threshold)
    if n_levels == 0:
        warnings.append("No cluster exceeds the greenness threshold => no algae detected")
 
    levels = np.zeros(valid.shape, np.uint8)
    levels[valid] = level_of_cluster[labels]
    levels = smooth_levels(levels, valid, cfg.smooth)
    levels = remove_small_blobs(levels, cfg.min_blob)
 
    stats = compute_stats(levels, valid, features, n_levels, masks, name, rgb.shape[:2], warnings)
 
    palette = level_palette(n_levels, cfg.colormap)
    overlay = make_overlay(rgb, levels, excluded, palette, cfg.opacity)
    return {
        "rgb": rgb, "stats": stats, "levels": levels, "masks": masks,
        "excluded": excluded, "overlay": overlay, "palette": palette,
    }
 
 
def analyze_image(path: Path, cfg: Config, out_dir: Path) -> dict:
    """Read an image from disk, analyze it and write the results to a folder."""
    rgb = load_image(path, cfg.max_width)
    res = analyze_rgb(rgb, cfg, str(path))
    stats, masks, overlay, palette = res["stats"], res["masks"], res["overlay"], res["palette"]
 
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    write_image(out_dir / f"{stem}_density.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    render_report(rgb, masks, overlay, stats, palette, out_dir / f"{stem}_report.png")
    with open(out_dir / f"{stem}_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def collect_inputs(p: Path) -> List[Path]:
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    return [p]
 
 
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Analyze algae density from images using K-means, with exclusion of noisy regions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, type=Path, help="Image file or folder of images")
    ap.add_argument("--output-dir", type=Path, default=Path("output"), help="Output folder")
    ap.add_argument("--clusters", type=int, default=5, help="Number of K-means clusters (2..12)")
    ap.add_argument("--algae-threshold", type=float, default=0.04,
                    help="Absolute ExG threshold for a cluster to count as algae "
                         "(raise if turbid water is mistaken for algae)")
    ap.add_argument("--colormap", choices=list(CMAPS), default="green")
    ap.add_argument("--opacity", type=float, default=0.7, help="Overlay opacity 0..1")
    ap.add_argument("--max-width", type=int, default=900, help="Maximum processing width (0 = keep original)")
    ap.add_argument("--denoise", type=int, default=1, help="Gaussian blur level before clustering (0 = off)")
    ap.add_argument("--smooth", type=int, default=5, help="Median filter size for label smoothing (0 = off)")
    ap.add_argument("--min-blob", type=int, default=30, help="Drop algae blobs smaller than this many pixels")
    ap.add_argument("--seed", type=int, default=42)
 
    g = ap.add_argument_group("Noise exclusion")
    g.add_argument("--no-glare-filter", action="store_true", help="Do not exclude glare/foam/clouds")
    g.add_argument("--no-shadow-filter", action="store_true", help="Do not exclude deep shadows")
    g.add_argument("--no-object-filter", action="store_true", help="Do not exclude red/purple objects")
    g.add_argument("--glare-v", type=int, default=235, help="Minimum Value for glare (0..255)")
    g.add_argument("--glare-s", type=int, default=40, help="Maximum Saturation for glare")
    g.add_argument("--shadow-v", type=int, default=35, help="Maximum Value for deep shadow")
    g.add_argument("--roi-mask", type=Path, help="ROI mask image (white = water area to analyze)")
    g.add_argument("--roi-polygon", type=str,
                   help="Polygon ROI, 0..1 coordinates: 'x1,y1;x2,y2;x3,y3;...'")
    g.add_argument("--exclude-mask", type=Path, help="Exclusion mask image (white = excluded)")
    g.add_argument("--min-valid", type=float, default=0.15, help="Minimum valid pixel fraction 0..1")
    return ap
 
 
def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
 
    if not 2 <= args.clusters <= 12:
        log.error("--clusters must be between 2 and 12")
        return 2
    if not 0.0 <= args.opacity <= 1.0:
        log.error("--opacity must be between 0 and 1")
        return 2
    if not args.input.exists():
        log.error("Not found: %s", args.input)
        return 1
 
    cfg = Config(
        clusters=args.clusters, max_width=args.max_width, denoise=args.denoise,
        algae_threshold=args.algae_threshold, seed=args.seed, smooth=args.smooth,
        min_blob=args.min_blob, min_valid_frac=args.min_valid,
        filter_glare=not args.no_glare_filter, filter_shadow=not args.no_shadow_filter,
        filter_objects=not args.no_object_filter, glare_v=args.glare_v, glare_s=args.glare_s,
        shadow_v=args.shadow_v, roi_mask=args.roi_mask, roi_polygon=args.roi_polygon,
        exclude_mask=args.exclude_mask, opacity=args.opacity, colormap=args.colormap,
    )
 
    files = collect_inputs(args.input)
    if not files:
        log.error("No valid images in: %s", args.input)
        return 1
 
    rows = []
    for f in files:
        log.info("Processing: %s", f.name)
        try:
            s = analyze_image(f, cfg, args.output_dir)
        except Exception as e:  # one failing image must not stop the whole batch
            log.error("  Skipping %s: %s", f.name, e)
            continue
        rows.append(s)
        log.info("  Algae cover %.1f%% | density index %.2f | %s",
                 s["algae_cover_pct"], s["density_index"], s["category"])
        for w in s["warnings"]:
            log.warning("  ! %s", w)
 
    if rows:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "summary.csv"
        cols = ["file", "valid_area_pct", "algae_cover_pct", "density_index",
                "mean_exg_valid", "mean_exg_algae", "n_algae_levels", "category"]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        log.info("Done: %d/%d images. Results in: %s", len(rows), len(files), args.output_dir)
    return 0 if rows else 1
 
 
if __name__ == "__main__":
    sys.exit(main())
 
