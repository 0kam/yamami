# Quick Start Guide

This guide walks you through a complete YAMAMI workflow from image ingestion
to phenology extraction.

## Overview

The YAMAMI pipeline consists of these main stages:

1. **Ingest**: Scan directory and create image index
2. **Profile**: Extract metadata and compute image statistics
3. **Segment**: Semantic segmentation (sky, cloud, fog, snow)
4. **Pre-QC**: Filter images before alignment
5. **Align**: Detect camera shifts and align images
6. **AOI**: Extract area of interest (skyline detection)
7. **QC**: Full quality control with AOI consideration
8. **Snow/Phenology**: Extract environmental metrics
9. **Viz**: Generate visualizations
10. **Export**: Package results with provenance

## Filename Convention

YAMAMI expects images following this naming pattern:

```
{site}_{azimuth}_{camera}_{band}_{YYYYMMDD}_{HHMM}.{ext}
```

Example: `MRD_N_NikL_RGB_20240501_0800.jpg`

- **site**: Site code (e.g., MRD)
- **azimuth**: Camera direction (N, S, E, W)
- **camera**: Camera identifier (e.g., NikL)
- **band**: Image type (RGB, NIR)
- **date**: Date in YYYYMMDD format
- **time**: Time in HHMM format

## Step-by-Step Workflow

### Step 1: Ingest Images

Scan a directory to create an index of all images:

```python
import yamami

# Scan directory recursively for image files
index = yamami.ingest("/path/to/images")
print(f"Found {len(index)} images")
```

### Step 2: Profile Images

Extract metadata and compute luminance statistics:

```python
# Profile all images
profile_df = yamami.profile(index)

# View profile columns
print(profile_df.columns.tolist())
# ['path', 'filename', 'site', 'azimuth', 'camera', 'band',
#  'timestamp', 'width', 'height', 'filesize', 'mean_luma',
#  'std_luma', 'p05_luma', 'p50_luma', 'p95_luma']
```

### Step 3: Semantic Segmentation

Segment images to identify sky, cloud, fog, and snow regions.
SAM3 requires a CUDA-capable GPU:

```python
# Prepare profile for segmentation (rename path to filepath)
segment_profile = profile_df.rename(columns={"path": "filepath"})

# Run segmentation (SAM3 requires CUDA)
labels, masks = yamami.segment(segment_profile, device="cuda")

# labels contains ratio of each class per image
# masks contains binary masks for each class
```

### Step 4: Pre-Alignment QC

Filter out unusable images before alignment:

```python
# Apply pre-alignment quality control
qc_result = yamami.pre_qc(segment_profile, labels)

# Filter to usable images
usable_mask = qc_result["is_usable"]
usable_profile = segment_profile[usable_mask].reset_index(drop=True)
print(f"{usable_mask.sum()}/{len(usable_mask)} images passed pre-QC")
```

### Step 5: Align Images

Detect camera shifts and align images to a reference:

```python
# Select reference image (best quality mid-season image)
ref_image = usable_profile["filepath"].iloc[len(usable_profile) // 2]

# Align all images
segments, warp_params, aligned_dir = yamami.align(
    usable_profile,
    ref=ref_image,
    output_dir="output/aligned"
)

print(f"Created {len(segments)} segments")
print(f"Aligned images saved to: {aligned_dir}")
```

### Step 6: Extract AOI

Detect the skyline and create an area of interest mask:

```python
# Extract AOI from reference image
skyline, aoi_mask = yamami.aoi(ref_image, output_dir="output/aoi")

# aoi_mask is a binary mask (0/255) where 255 = vegetation area
print(f"AOI covers {(aoi_mask > 0).sum()} pixels")
```

### Step 7: Full QC with AOI

Apply full quality control considering AOI cloud coverage:

```python
# Add AOI cloud ratio to labels (computed from masks and AOI)
labels["aoi_cloud_ratio"] = 0.1  # Example value

# Full QC
final_qc = yamami.qc(usable_profile, labels, aoi_mask)
final_profile = usable_profile[final_qc["is_usable"]]
```

### Step 8a: Snow Analysis

Detect snow and estimate snowmelt dates:

```python
# Prepare profile with RGB values
snow_profile = final_profile.copy()
snow_profile["R_mean"] = 180  # Example values
snow_profile["G_mean"] = 180
snow_profile["B_mean"] = 180
snow_profile["doy"] = final_profile["timestamp"].dt.dayofyear

# Detect snow
snow_masks, thresholds = yamami.snow(snow_profile, aoi_mask)

# Estimate snowmelt DOY per pixel
snowmelt_df = yamami.snowmelt(snow_masks, aoi_mask)
print(snowmelt_df.columns.tolist())
# ['row', 'col', 'doy', 'confidence', 'status']
```

### Step 8b: Phenology Analysis

Calculate greenness ratio and extract phenological dates:

```python
# Compute GR time series
daily_images, timeseries = yamami.gr(final_profile, aoi_mask > 0)

# Extract phenology dates
pheno_result = yamami.phenology(timeseries)
print(pheno_result[["gup_doy", "gdown_doy", "gmax_doy"]])
```

### Step 9: Visualization

Generate color-coded DOY maps:

```python
import pandas as pd

# Create phenology data for visualization
pheno_data = pd.DataFrame({
    "row": [0, 0, 1, 1],
    "col": [0, 1, 0, 1],
    "gup": [100, 110, 105, 115],
    "gdown": [250, 260, 255, 265],
    "gmax": [175, 185, 180, 190],
})

# Generate visualizations
viz_result = yamami.viz(pheno_data, output_dir="output/viz")
print(f"Created: {list(viz_result.keys())}")
# ['gup', 'gdown', 'gmax']
```

### Step 10: Export Results

Package all results with provenance tracking:

```python
# Export all results
export_path = yamami.export("output", export_dir="export")
print(f"Results exported to: {export_path}")

# Export includes:
# - processing_log.json with version, timestamp, file hashes
# - All CSV data files
# - All visualization PNGs
```

## Complete Example Script

```python
"""
Complete YAMAMI workflow example.
"""
import yamami

# Configuration
IMAGE_DIR = "/path/to/images"
OUTPUT_DIR = "output"

# 1. Ingest and profile
index = yamami.ingest(IMAGE_DIR)
profile = yamami.profile(index)
profile = profile.rename(columns={"path": "filepath"})

# 2. Segment and pre-QC
labels, masks = yamami.segment(profile)
qc1 = yamami.pre_qc(profile, labels)
profile = profile[qc1["is_usable"]].reset_index(drop=True)

# 3. Align
ref = profile["filepath"].iloc[len(profile) // 2]
segments, warp_params, aligned_dir = yamami.align(
    profile, ref=ref, output_dir=f"{OUTPUT_DIR}/aligned"
)

# 4. AOI and full QC
skyline, aoi_mask = yamami.aoi(ref, output_dir=f"{OUTPUT_DIR}/aoi")
labels = labels[qc1["is_usable"]].reset_index(drop=True)
labels["aoi_cloud_ratio"] = 0.1
qc2 = yamami.qc(profile, labels, aoi_mask)
profile = profile[qc2["is_usable"]].reset_index(drop=True)

# 5. Phenology
daily_images, timeseries = yamami.gr(profile, aoi_mask > 0)
pheno = yamami.phenology(timeseries)

# 6. Visualize and export
yamami.viz(pheno, output_dir=f"{OUTPUT_DIR}/viz")
export_path = yamami.export(OUTPUT_DIR)

print(f"Complete! Results in: {export_path}")
```

## Next Steps

- See the [API Reference](api/index.rst) for detailed function documentation
- Customize QC thresholds for your site conditions
- Explore pixel-level phenology mapping
