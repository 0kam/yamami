# Quickstart Guide: Yamami

**Date**: 2026-01-27
**Estimated time**: 30 minutes

## Prerequisites

- Python 3.12
- pip
- CUDA-compatible GPU (required for SAM3 segmentation)

## Installation

```bash
pip install yamami
```

Verify installation:

```python
import yamami
print(yamami.__version__)
```

## Prepare Your Data

### Directory Structure

Organize your time-lapse images in a single directory:

```
data/
└── raw/
    ├── MRD_N_NikL_RGB_20240501_0800.jpg
    ├── MRD_N_NikL_RGB_20240501_0900.jpg
    ├── MRD_N_NikL_RGB_20240501_1000.jpg
    └── ... (thousands of images)
```

### Naming Convention

Images must follow this pattern:
```
{site}_{azimuth}_{camera}_{band}_{YYYYMMDD}_{HHMM}.jpg
```

Example: `MRD_N_NikL_RGB_20240501_0800.jpg`
- Site: MRD
- Azimuth: N (north)
- Camera: NikL
- Band: RGB
- Date: 2024-05-01
- Time: 08:00

## Basic Workflow

```python
import yamami

# Step 1: Ingest Images
# Scan your image directory
index = yamami.ingest("data/raw/")
print(f"Found {len(index)} images")

# Step 2: Profile Images
# Compute metadata and statistics
profile = yamami.profile(index)
print(f"Profiled {len(profile)} images")

# Step 3: Segment Images
# Run semantic segmentation (this may take time)
labels, masks = yamami.segment(
    profile,
    prompts=["sky", "cloud", "fog", "snow"],
    resize=512,
    device="cuda"
)

# Step 4: Pre-QC
# Initial quality filtering
profile_pre = yamami.pre_qc(
    profile,
    labels,
    brightness_min=0.08,
    fog_max=0.30,
    cloud_max=0.40
)
print(f"Usable images after pre-QC: {profile_pre['is_usable'].sum()}")

# Step 5: Select Reference Image
# Choose a clear, well-exposed image as reference
# Typically a mid-summer image with clear sky and good visibility
ref_image = "data/raw/MRD_N_NikL_RGB_20240715_1200.jpg"

# Step 6: Align Images
# Detect camera shifts and align to reference
segments, warp_params, aligned_dir = yamami.align(
    profile_pre,
    ref=ref_image,
    method="roma",
    max_rounds=3
)
print(f"Detected {len(segments)} camera position segments")

# Step 7: Extract AOI
# Generate analysis-of-interest mask from reference
skyline, aoi_mask = yamami.aoi(ref_image)

# Step 8: Full QC
# Quality control with AOI consideration
profile_qc = yamami.qc(
    profile,
    labels,
    aoi_mask,
    aoi_cloud_max=0.20
)
print(f"Final usable images: {profile_qc['is_usable'].sum()}")

# Step 9: Extract Snow
# Classify snow/non-snow
snow_masks, thresholds = yamami.snow(
    profile_qc,
    aoi_mask,
    otsu=True
)

# Step 10: Estimate Snowmelt
# Calculate snowmelt day-of-year
snowmelt_map = yamami.snowmelt(snow_masks, aoi_mask)

# Step 11: Compute Greenness
# Calculate greenness ratio time series
gr_daily, gr_timeseries = yamami.gr(
    profile_qc,
    aoi_mask,
    daily_max=True
)

# Step 12: Fit Phenology
# Extract phenology dates
phenology_results = yamami.phenology(
    gr_daily,
    model="double_sigmoid"
)
print(f"Green-up DOY range: {phenology_results['gup_doy'].min()}-{phenology_results['gup_doy'].max()}")

# Step 13: Visualize
# Generate color-coded maps
viz_outputs = yamami.viz(
    phenology_results,
    colormap="viridis"
)

# Step 14: Export
# Organize all results
yamami.export(
    output_dir="outputs/",
    export_dir="outputs/export/"
)
```

## Output Files

After running the workflow, you'll have:

```
outputs/
├── images/
│   ├── index.csv           # Image catalog
│   ├── profile.csv         # Image metadata and statistics
│   └── profile_qc.csv      # QC results
├── seg/
│   ├── labels.csv          # Mask areas
│   └── masks/              # PNG mask files
├── align/
│   ├── segments.csv        # Camera shift segments
│   ├── warp_params.csv     # Transformation parameters
│   └── aligned/            # Aligned images
├── aoi/
│   ├── skyline.csv         # Skyline coordinates
│   └── aoi_mask.png        # Binary mask
├── snow/
│   ├── extract/            # Binary snow masks
│   └── thresholds.csv      # Threshold values
├── snowmelt/
│   ├── doy_map.png         # Snowmelt timing map
│   └── doy_map.csv         # Pixel-level data
├── gr/
│   ├── daily_max/          # Daily maximum GR images
│   └── timeseries.csv      # Time series data
├── phenology/
│   ├── gup.png             # Green-up map
│   ├── gdown.png           # Green-down map
│   └── phenology.csv       # Pixel-level dates
├── viz/                    # Publication-ready visualizations
└── export/                 # Organized results and processing log
```

## Troubleshooting

### "No images found"
- Check directory path
- Verify image extensions (.jpg, .jpeg, .tif, .tiff)

### "Parse failed" for many images
- Check filename pattern matches expected format
- Verify timestamps are valid dates

### Alignment failures
- Try different matching method: `method="sift"` or `method="loftr"`
- Check reference image quality
- Increase max rounds: `max_rounds=5`

### Out of memory
- Reduce segment batch size
- Use smaller inference resolution: `resize=256`

## Next Steps

- Read the [API documentation](https://yamami.readthedocs.io/en/latest/api/)
- See [example notebooks](https://github.com/your-org/yamami/tree/main/examples)
- Report issues on [GitHub](https://github.com/your-org/yamami/issues)
