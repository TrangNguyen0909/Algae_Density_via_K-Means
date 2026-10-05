# Algae Density Analysis

Estimate algae density from a photo of a water surface. The tool excludes unreliable regions (glare, shadows, boats, shoreline), groups the remaining pixels with K-means, and reports how much of the water is covered by algae and how dense it is.

## Features

- **Noise exclusion**: glare/foam/clouds, deep shadows, red/purple objects, user-defined ROI, manual exclusion mask.
- **Absolute algae threshold**: an image with no algae is reported as "No algae detected" instead of being forced into density levels.
- **Illumination-robust features**: chromaticity ExG, Lab a\*, Saturation, Value.
- **Two interfaces**: command line (single image or whole folder) and a Streamlit web app.
- **Outputs**: density map, 4-panel report, JSON statistics, CSV summary for batches.

## Project structure

```
algae_density.py   Core logic + command-line interface
app.py             Streamlit web interface (imports algae_density.py)
requirements.txt   Python dependencies
```

Keep `app.py` and `algae_density.py` in the same folder.

## Installation

Python 3.9 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Usage

### Web app

```bash
streamlit run app.py
```

Open http://localhost:8501, upload an image, and adjust the sidebar settings. The Excluded Regions panel shows which pixels were ignored.

### Command line

```bash
# Single image
python algae_density.py --input image.jpg

# A whole folder (also writes summary.csv)
python algae_density.py --input photos/ --output-dir results

# Restrict analysis to the water surface (polygon, 0..1 coordinates)
python algae_density.py --input image.jpg --roi-polygon "0,0.4;1,0.4;1,1;0,1"

# Use mask images (white = area of interest / area to exclude)
python algae_density.py --input image.jpg --roi-mask water.png --exclude-mask boats.png
```

## How it works

1. **Preprocess**: read, resize (default max width 900 px), light Gaussian blur.
2. **Exclude** unreliable pixels. The remaining pixels are the "valid area"; all percentages use it as the denominator.
3. **Features** per pixel: chromaticity ExG (`2g' - r' - b'`), Lab `-a*`, Saturation, Value.
4. **K-means** on valid pixels only (fitted on a 200k-pixel sample, then applied to all pixels).
5. **Levels**: each cluster's mean ExG is compared with the algae threshold. Below it = Level 0 (water / no algae). Above it = Levels 1..m, ordered by increasing ExG.
6. **Post-process**: median smoothing, removal of tiny blobs, then statistics and export.

## Parameters

### Main

| Option | Default | Description |
|---|---|---|
| `--input` | required | Image file or folder |
| `--output-dir` | `output` | Where results are written |
| `--clusters` | 5 | K-means clusters (2..12) |
| `--algae-threshold` | 0.04 | Minimum ExG for a cluster to count as algae. Raise if turbid water is flagged; lower if sparse algae is missed |
| `--colormap` | `green` | `green` or `thermal` |
| `--opacity` | 0.7 | Overlay opacity (0..1) |
| `--max-width` | 900 | Processing width (0 = original size) |
| `--denoise` | 1 | Gaussian blur level (0 = off) |
| `--smooth` | 5 | Median filter size (0 = off) |
| `--min-blob` | 30 | Remove algae blobs smaller than this many pixels |
| `--seed` | 42 | Random seed |

### Exclusion

| Option | Default | Description |
|---|---|---|
| `--no-glare-filter` | off | Keep glare/foam/clouds |
| `--glare-v` / `--glare-s` | 235 / 40 | Glare = Value >= glare-v and Saturation <= glare-s |
| `--no-shadow-filter` | off | Keep deep shadows |
| `--shadow-v` | 35 | Shadow = Value <= shadow-v |
| `--no-object-filter` | off | Keep red/purple objects |
| `--roi-polygon` | none | `"x1,y1;x2,y2;x3,y3"`, coordinates 0..1, origin at top-left |
| `--roi-mask` | none | Mask image, white = analyze |
| `--exclude-mask` | none | Mask image, white = exclude |
| `--min-valid` | 0.15 | Minimum valid-pixel fraction; below this the image is rejected |

If both `--roi-polygon` and `--roi-mask` are given, the analyzed area is their intersection.

## Outputs

For each image `name.jpg`:

| File | Content |
|---|---|
| `name_density.png` | Original image with the density overlay (excluded areas shown gray) |
| `name_report.png` | Original, excluded regions, density map, distribution chart |
| `name_stats.json` | All statistics and warnings |
| `summary.csv` | One row per image (batch mode) |

Key statistics:

| Field | Meaning |
|---|---|
| `algae_cover_pct` | % of valid area with algae (Level >= 1) |
| `density_index` | 0..1, mean level divided by maximum level over the valid area |
| `mean_exg_valid` | Mean greenness of the valid area; **comparable across images** |
| `category` | Negligible / Low / Moderate / High / Very high, from cover % (<1, <10, <30, <60, >=60) |
| `level_pct` | % of valid area per level (index 0 = no algae / water) |

## Limitations

- **Shoreline vegetation looks like algae.** Always define an ROI covering the water only when trees or grass appear in the photo.
- **Density levels are only comparable within one image.** Cluster centers differ per image. Compare images using `mean_exg_valid` and `algae_cover_pct`.
- **Thresholds are empirical.** The 0.04 threshold and the category cut-offs are starting values. Calibrate them against field data (for example chlorophyll-a) before drawing scientific conclusions.
- **Lighting and angle matter.** Use consistent conditions when monitoring over time. Avoid heavy glare.
- Results come from color only; the tool cannot identify algae species or measure biomass.

## Troubleshooting

| Problem | Fix |
|---|---|
| Clean water shows algae | Raise `--algae-threshold` (for example 0.06-0.10) |
| Sparse algae not detected | Lower `--algae-threshold` (for example 0.02) |
| Result looks speckled | Increase `--denoise`, `--smooth` or `--min-blob` |
| "Only X% of pixels remain valid" | Loosen exclusion filters or enlarge the ROI |
| Normal water is marked as excluded | Disable the matching filter or adjust its thresholds |
| Trees/grass are tinted as algae | Define an ROI (`--roi-polygon` or `--roi-mask`) |
| Streamlit warns about `use_container_width` | Use Streamlit >= 1.50 (`pip install -U streamlit`) |
