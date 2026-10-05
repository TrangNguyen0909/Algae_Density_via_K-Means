import io
import json
import tempfile
from pathlib import Path
 
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from algae_density import (
    CMAPS,
    EXCL_STYLE,
    Config,
    analyze_rgb,
    level_labels,
    level_palette,
    make_exclusion_view,
    make_overlay,
    render_report,
    resize_rgb,
)
 
COLOR_NAME = {
    "glare": "yellow", "shadow": "blue", "object": "red",
    "outside_roi": "gray", "user_exclude": "magenta",
}
 
st.set_page_config(page_title="Algae Density Analysis", layout="wide")
 
st.title("Image-Based Algae Density Analysis Tool")
st.caption(
    "Upload an image of a water area containing algae. The tool excludes noisy regions "
    "(glare, deep shadows, red/purple objects, areas outside the ROI), then uses K-means "
    "to split the remaining area into density levels. Darker colors mean higher density; "
    "uncolored areas are water / no algae."
)
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _tmp_file(data: bytes, suffix: str) -> Path:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(data)
    f.close()
    return Path(f.name)
 
 
@st.cache_data(show_spinner=False)
def run_analysis(img_bytes: bytes, params: dict, roi_bytes, excl_bytes) -> dict:
    """
    The heavy part (exclusion + K-means). Cached, so changing the color map or
    overlay opacity does not re-run K-means.
    """
    pil = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    rgb = resize_rgb(np.array(pil), params["max_width"])
 
    tmp_files = []
    try:
        roi_path = excl_path = None
        if roi_bytes:
            roi_path = _tmp_file(roi_bytes, ".png")
            tmp_files.append(roi_path)
        if excl_bytes:
            excl_path = _tmp_file(excl_bytes, ".png")
            tmp_files.append(excl_path)
        cfg = Config(**params, roi_mask=roi_path, exclude_mask=excl_path)
        return analyze_rgb(rgb, cfg, "uploaded")
    finally:
        for p in tmp_files:
            p.unlink(missing_ok=True)
 
 
# ---------------------------------------------------------------------------
# Sidebar: parameters
# ---------------------------------------------------------------------------
 
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"]
    )
 
    n_clusters = st.slider("Number of K-means clusters", min_value=2, max_value=8, value=5)
 
    colormap_name = st.selectbox(
        "Color map", list(CMAPS), index=0,
        format_func=lambda x: "Green" if x == "green" else "Thermal",
    )
    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.7, 0.05)
    max_width = st.slider("Maximum processing width (px)", 300, 1600, 900, 50)
    denoise = st.slider("Denoising level (Gaussian blur)", 0, 5, 1)
 
    with st.expander("Algae threshold & smoothing"):
        algae_threshold = st.slider(
            "Greenness threshold (ExG)", 0.0, 0.30, 0.04, 0.005,
            help="Clusters whose mean ExG is below this threshold are treated as water / no algae. "
                 "Raise it if turbid water is mistaken for algae; lower it if sparse algae is missed.",
        )
        smooth = st.slider("Label smoothing (median, 0 = off)", 0, 9, 5)
        min_blob = st.slider("Drop algae blobs smaller than (pixels)", 0, 500, 30, 5)
 
    with st.expander("Noise exclusion"):
        filter_glare = st.checkbox("Exclude glare / foam / clouds", value=True)
        glare_v = st.slider("Glare: minimum Value", 150, 255, 235, disabled=not filter_glare)
        glare_s = st.slider("Glare: maximum Saturation", 0, 120, 40, disabled=not filter_glare)
        filter_shadow = st.checkbox("Exclude deep shadows", value=True)
        shadow_v = st.slider("Shadow: maximum Value", 0, 100, 35, disabled=not filter_shadow)
        filter_objects = st.checkbox("Exclude red / purple objects", value=True)
 
    with st.expander("Region of interest (ROI) & manual exclusion"):
        roi_polygon = st.text_input(
            "Polygon ROI (0..1 coordinates)",
            placeholder="0.1,0.3;0.9,0.3;0.9,0.95;0.1,0.95",
            help="'x,y' points separated by ';', origin at the top-left corner of the image.",
        )
        roi_file = st.file_uploader("ROI mask (white = water area)", type=["png", "jpg", "jpeg"], key="roi")
        excl_file = st.file_uploader("Exclusion mask (white = excluded)", type=["png", "jpg", "jpeg"], key="excl")
        st.caption("Green trees and shoreline grass are easily mistaken for algae; "
                   "define an ROI that covers the water surface only.")
 
 
# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
 
if uploaded_file is None:
    st.info("No image has been uploaded yet. Please select an image from the sidebar to get started.")
    st.stop()
 
params = dict(
    clusters=n_clusters, max_width=max_width, denoise=denoise,
    algae_threshold=algae_threshold, smooth=smooth, min_blob=min_blob,
    filter_glare=filter_glare, filter_shadow=filter_shadow, filter_objects=filter_objects,
    glare_v=glare_v, glare_s=glare_s, shadow_v=shadow_v,
    roi_polygon=roi_polygon.strip() or None,
)
 
try:
    with st.spinner("Analyzing..."):
        res = run_analysis(
            uploaded_file.getvalue(), params,
            roi_file.getvalue() if roi_file else None,
            excl_file.getvalue() if excl_file else None,
        )
except ValueError as e:
    st.error(f"Analysis failed: {e}")
    st.stop()
 
rgb, stats, masks, excluded = res["rgb"], res["stats"], res["masks"], res["excluded"]
levels = res["levels"]
n_levels = stats["n_algae_levels"]
 
# Overlay is recomputed for the current color map / opacity (no K-means re-run needed)
palette = level_palette(n_levels, colormap_name)
overlay = make_overlay(rgb, levels, excluded, palette, opacity)
 
for w in stats["warnings"]:
    st.warning(w)
 
m1, m2, m3, m4 = st.columns(4)
m1.metric("Algae cover", f"{stats['algae_cover_pct']:.1f}%")
m2.metric("Density index (0-1)", f"{stats['density_index']:.2f}")
m3.metric("Rating", stats["category"])
m4.metric("Valid area", f"{stats['valid_area_pct']:.1f}%")
 
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Original Image")
    st.image(rgb, width="stretch")
with col2:
    st.subheader("Excluded Regions")
    st.image(make_exclusion_view(rgb, masks), width="stretch")
    if masks:
        st.caption(" · ".join(
            f"{COLOR_NAME[k]}: {EXCL_STYLE[k][1]} ({stats['excluded_pct_by_type'][k]:.1f}%)"
            for k in EXCL_STYLE if k in masks
        ))
    else:
        st.caption("No regions were excluded.")
with col3:
    st.subheader("Algae Density Map")
    st.image(overlay, width="stretch")
 
st.subheader("Area Distribution by Density Level")
labels = level_labels(n_levels)
pct = stats["level_pct"]
bar_colors = [np.array([158, 202, 225]) / 255.0] + [palette[i] / 255.0 for i in range(1, n_levels + 1)]
 
fig, ax = plt.subplots(figsize=(8, max(2.5, 0.45 * len(pct) + 1)))
y_pos = np.arange(len(pct))
ax.barh(y_pos, pct, color=bar_colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Share of valid area (%)")
ax.set_xlim(0, max(pct) * 1.18 + 1)
for i, v in enumerate(pct):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
 
with st.expander("Detailed table"):
    st.dataframe(
        pd.DataFrame({
            "Level": labels,
            "Valid area (%)": pct,
            "Mean ExG": stats["level_mean_exg"],
        }),
        width="stretch", hide_index=True,
    )
    st.caption(
        f"Mean ExG over the whole valid area: {stats['mean_exg_valid']} "
        "(this value is comparable across images; density levels are only comparable within one image)."
    )
 
# Downloads
st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
 
buf = io.BytesIO()
Image.fromarray(overlay).save(buf, format="PNG")
d1.download_button("Density Map (PNG)", buf.getvalue(), "algae_density_map.png", "image/png")
 
report_buf = io.BytesIO()
render_report(rgb, masks, overlay, stats, palette, report_buf)
d2.download_button("Full Report (PNG)", report_buf.getvalue(), "algae_report.png", "image/png")
 
d3.download_button(
    "Statistics (JSON)",
    json.dumps(stats, ensure_ascii=False, indent=2).encode("utf-8"),
    "algae_stats.json", "application/json",
)
 
