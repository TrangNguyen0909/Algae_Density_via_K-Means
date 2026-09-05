"""
Web interface for the algae density analysis tool (reuses the K-means logic
from analyze_algae.py). Allows users to upload images directly, adjust
parameters using sliders, view the results, and download the output.

How to run:
    streamlit run app.py

The browser will then open automatically at http://localhost:8501
"""

import io

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_algae import compute_features, cluster_density, build_colormap


st.set_page_config(page_title="Algae Density Analysis", layout="wide")

st.title("Image-Based Algae Density Analysis Tool")
st.caption(
    "Upload an image of a water area containing algae. The tool uses K-means "
    "(unsupervised machine learning) to visualize algae density: darker colors "
    "represent higher density, while lighter colors represent lower density."
)

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    n_clusters = st.slider(
        "Number of density levels",
        min_value=2,
        max_value=8,
        value=5
    )

    colormap_name = st.selectbox(
        "Color map",
        ["green", "thermal"],
        index=0,
        format_func=lambda x: "Green" if x == "green" else "Thermal"
    )

    opacity = st.slider(
        "Overlay opacity",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05
    )

    max_width = st.slider(
        "Maximum processing width (px)",
        min_value=300,
        max_value=1600,
        value=900,
        step=50
    )

    denoise = st.slider(
        "Denoising level (Gaussian blur)",
        min_value=0,
        max_value=5,
        value=1
    )


def resize_if_needed(rgb: np.ndarray, max_width: int) -> np.ndarray:
    h, w = rgb.shape[:2]

    if w > max_width:
        scale = max_width / w
        img = Image.fromarray(rgb).resize(
            (max_width, int(h * scale))
        )
        return np.array(img)

    return rgb


if uploaded_file is None:
    st.info(
        "No image has been uploaded yet. "
        "Please select an image from the sidebar to get started."
    )
else:
    pil_img = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_img)
    rgb = resize_if_needed(rgb, max_width)

    with st.spinner("Analyzing..."):
        features = compute_features(rgb, denoise)
        density_rank, _ = cluster_density(features, n_clusters)

        cmap = build_colormap(colormap_name)

        norm_rank = density_rank.astype(np.float32) / max(
            1, n_clusters - 1
        )

        heat_rgba = cmap(norm_rank)
        heat_rgb = (heat_rgba[..., :3] * 255).astype(np.float32)

        blended = (
            (1 - opacity) * rgb.astype(np.float32)
            + opacity * heat_rgb
        )

        blended = np.clip(
            blended, 0, 255
        ).astype(np.uint8)

        counts = np.array([
            (density_rank == i).sum()
            for i in range(n_clusters)
        ])

        percentages = 100.0 * counts / counts.sum()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(rgb, use_container_width=True)

    with col2:
        st.subheader("Algae Density Map")
        st.image(blended, use_container_width=True)

    st.subheader("Area Distribution by Density Level")

    fig, ax = plt.subplots(figsize=(8, 2.5))

    labels = [
        f"Level {i + 1}"
        for i in range(n_clusters)
    ]

    bar_colors = [
        cmap(i / max(1, n_clusters - 1))
        for i in range(n_clusters)
    ]

    y_pos = np.arange(n_clusters)

    ax.barh(
        y_pos,
        percentages,
        color=bar_colors
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Area Percentage (%)")

    for i, v in enumerate(percentages):
        ax.text(
            v + 0.5,
            i,
            f"{v:.1f}%",
            va="center",
            fontsize=9
        )

    fig.tight_layout()
    st.pyplot(fig)

    out_img = Image.fromarray(blended)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")

    st.download_button(
        label="Download Density Map (PNG)",
        data=buf.getvalue(),
        file_name="algae_density_map.png",
        mime="image/png",
    )