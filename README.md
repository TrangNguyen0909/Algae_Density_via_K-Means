# Algae Density Analysis Tool — Python + K-means

An image-based tool for analyzing and visualizing **algae density** using Python and **K-means clustering (unsupervised machine learning)**.

Instead of applying a fixed color threshold to individual pixels, the system combines three color features — **Excess Green, Saturation, and Value** — to identify density levels adaptively for each input image.

## Key Features

* K-means-based image segmentation.
* Combines three color features for more robust classification.
* Gaussian Blur for noise reduction.
* Classifies algae density from **low to high**.
* Generates a visual algae density map.
* Calculates area percentage for each density level.
* Streamlit web interface for easy image upload and parameter adjustment.

## Installation

Requires Python 3.9 or higher.

```bash
cd algae-density-tool-python
pip install -r requirements.txt
```

## Web Interface

The recommended way to use the tool is through the Streamlit interface:

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

The interface allows you to:

* Upload an image directly.
* Adjust the number of density levels.
* Select a color map.
* Adjust overlay opacity.
* Control the denoising level.
* View the original image and density map.
* View area statistics by density level.
* Download the processed result.

## Command-Line Usage

The tool can also be run without the web interface:

```bash
python analyze_algae.py --input input.jpg
```

The default output files are:

* `density_map.png` — algae density map.
* `analysis_report.png` — visual analysis report with area statistics.

### Advanced Options

```bash
python analyze_algae.py \
    --input input.jpg \
    --output result.png \
    --report report.png \
    --clusters 5 \
    --colormap green \
    --opacity 0.75 \
    --max-width 900 \
    --denoise 1
```

| Option        | Description                          | Default               |
| ------------- | ------------------------------------ | --------------------- |
| `--input`     | Path to the input image              | Required              |
| `--output`    | Path to save the density map         | `density_map.png`     |
| `--report`    | Path to save the visual report       | `analysis_report.png` |
| `--clusters`  | Number of density levels             | `5`                   |
| `--colormap`  | Color map: `green` or `thermal`      | `green`               |
| `--opacity`   | Density-map overlay opacity (0–1)    | `0.75`                |
| `--max-width` | Maximum image width for processing   | `900`                 |
| `--denoise`   | Gaussian blur level; `0` disables it | `1`                   |

## Processing Pipeline

```text
Input Image
     ↓
Image Preprocessing
     ↓
Color Feature Extraction
     ↓
Feature Standardization
     ↓
K-means Clustering
     ↓
Cluster Ranking by Excess Green
     ↓
Algae Density Map
     ↓
Area Statistics & Visual Report
```

### Feature Extraction

For each pixel, the system calculates three features:

* **Excess Green (2G − R − B)** — measures green intensity.
* **Saturation (HSV)** — measures color intensity.
* **Value (HSV)** — represents image brightness.

Using multiple features helps K-means distinguish algae-related regions more effectively than using a single color channel.

## Important Note

This tool is a **visual image-analysis system**, not a scientifically calibrated model for measuring algae biomass or concentration.

The results are primarily intended for **relative comparison between regions within the same image**. For scientific or quantitative applications, the results should be validated against field measurements or laboratory samples.
