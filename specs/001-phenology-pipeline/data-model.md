# Data Model: Yamami Phenology Pipeline

**Date**: 2026-01-27
**Branch**: 001-phenology-pipeline

## Terminology Note

This project uses "segment" in two distinct contexts:
- **SegmentationResult**: SAM3 semantic segmentation output (sky/cloud/fog/snow masks)
- **AlignmentSegment** (or just "Segment" in align.py): A contiguous time period with stable camera position

In code, these are clearly separated by module: `segment.py` handles SAM3 segmentation, while `align.py` handles alignment segments.

## Entity Definitions

### 1. Image

Represents a single camera capture from the time-lapse series.

**Attributes**:
| Field | Type | Description |
|-------|------|-------------|
| path | str | Absolute path to image file |
| filename | str | Basename of image file |
| site | str | Site identifier (e.g., "MRD") |
| azimuth | str | Camera azimuth/direction |
| camera | str | Camera identifier (e.g., "NikL") |
| band | str | Spectral band (e.g., "RGB") |
| timestamp | datetime | Capture timestamp from filename |
| width | int | Image width in pixels |
| height | int | Image height in pixels |
| filesize | int | File size in bytes |
| exif_ok | bool | Whether EXIF was successfully read |

**Identity**: `path` is unique
**Lifecycle**: Created during ingest, immutable thereafter

### 2. Profile

Metadata and quality metrics computed from an image.

**Attributes**:
| Field | Type | Description |
|-------|------|-------------|
| path | str | FK to Image |
| mean_luma | float | Mean luminance (0-1) |
| std_luma | float | Luminance standard deviation |
| p05_luma | float | 5th percentile luminance |
| p50_luma | float | Median luminance |
| p95_luma | float | 95th percentile luminance |
| sat_ratio | float | Saturation ratio (0-1) |
| parse_status | str | "ok" or error code |

**Identity**: `path` is unique (1:1 with Image)
**Lifecycle**: Created during profile stage

### 3. SegmentationResult

Segmentation mask areas for an image.

**Attributes**:
| Field | Type | Description |
|-------|------|-------------|
| path | str | FK to Image |
| mask_sky_area | float | Sky mask area ratio (0-1) |
| mask_cloud_area | float | Cloud mask area ratio (0-1) |
| mask_fog_area | float | Fog mask area ratio (0-1) |
| mask_snow_area | float | Snow mask area ratio (0-1) |
| mask_sky_path | str | Path to sky mask PNG |
| mask_cloud_path | str | Path to cloud mask PNG |
| mask_fog_path | str | Path to fog mask PNG |
| mask_snow_path | str | Path to snow mask PNG |

**Identity**: `path` is unique
**Lifecycle**: Created during segment stage

### 4. QCResult

Quality control assessment for an image.

**Attributes**:
| Field | Type | Description |
|-------|------|-------------|
| path | str | FK to Image |
| is_usable | bool | Whether image passes QC |
| reason_codes | list[str] | List of rejection reason codes |
| brightness | float | Computed brightness score |
| contrast | float | Computed contrast score |
| fog_score | float | Fog coverage score |
| cloud_score | float | Cloud coverage score |
| cloud_aoi_ratio | float | Cloud coverage within AOI |

**Reason codes**:
- `TOO_DARK`: Luminance below threshold
- `TOO_BRIGHT`: Luminance above threshold (saturated)
- `HIGH_FOG`: Fog coverage exceeds threshold
- `HIGH_CLOUD`: Cloud coverage exceeds threshold
- `AOI_OBSCURED`: AOI cloud ratio exceeds threshold
- `PARSE_FAILED`: Filename parsing failed

**Identity**: `path` is unique
**Lifecycle**: Created during qc stage

### 5. Segment (Alignment Segment)

A contiguous time period with stable camera position.

**Attributes**:
| Field | Type | Description |
|-------|------|-------------|
| segment_id | int | Unique segment identifier |
| start_ts | datetime | Segment start timestamp |
| end_ts | datetime | Segment end timestamp |
| n_frames | int | Number of frames in segment |
| anchor_path | str | FK to best-quality image in segment |
| ref_path | str | FK to reference image used for alignment |
| status | str | Alignment status |
| H_to_global | list[float] | 3x3 homography matrix (flattened) |
| reproj_err | float | Reprojection error in pixels |
| dx_med | float | Median x-displacement in segment |
| dy_med | float | Median y-displacement in segment |
| peak_med | float | Median phase correlation peak |

**Status codes**:
- `aligned`: Successfully aligned to reference
- `failed`: Alignment failed after max retries
- `pending`: Not yet processed

**Identity**: `segment_id` is unique
**Lifecycle**: Created during align stage, updated during iterative alignment

### 6. WarpParams

Geometric transformation for a single image.

**Attributes**:
| Field | Type | Description |
|-------|------|-------------|
| path | str | FK to Image |
| segment_id | int | FK to Segment |
| dx | float | X translation (pixels) |
| dy | float | Y translation (pixels) |
| rotation_deg | float | Rotation angle (degrees) |
| scale | float | Scale factor |
| status | str | "ok", "failed", or "skipped" |
| method | str | Matching method used |

**Identity**: `path` is unique
**Lifecycle**: Created during align stage

### 7. Skyline

Skyline coordinates extracted from reference image.

**Attributes**:
| Field | Type | Description |
|-------|------|-------------|
| x | int | Column index |
| y | int | Row index of skyline at column x |
| confidence | float | Detection confidence |

**Identity**: Composite (reference_image, x)
**Lifecycle**: Created during aoi stage

### 8. SnowmeltMap

Pixel-level snowmelt day-of-year values.

**Attributes**:
| Field | Type | Description |
|-------|------|-------------|
| row | int | Pixel row index |
| col | int | Pixel column index |
| doy | int | Day of year when snow disappeared (1-365) |
| confidence | float | Estimation confidence |
| status | str | "ok", "rock", "no_data" |

**Status codes**:
- `ok`: Valid snowmelt date estimated
- `rock`: High-reflectance rock (excluded)
- `no_data`: Insufficient observations

**Identity**: Composite (row, col)
**Storage**: As raster image (PNG/TIFF) with CSV metadata

### 9. PhenologyResult

Phenology dates for a single pixel.

**Attributes**:
| Field | Type | Description |
|-------|------|-------------|
| row | int | Pixel row index |
| col | int | Pixel column index |
| gup_doy | int | Green-up day of year |
| gdown_doy | int | Green-down day of year |
| gmax_doy | int | Maximum greenness day of year |
| gmax_value | float | Maximum GR value |
| fit_rmse | float | Sigmoid fit RMSE |
| fit_status | str | "ok" or "failed" |

**Identity**: Composite (row, col)
**Storage**: As raster images (PNG/TIFF) with CSV metadata

## Relationships

```
Image (1) ──── (1) Profile
  │
  ├──── (1) SegmentationResult
  │
  ├──── (1) QCResult
  │
  └──── (1) WarpParams ──── (N:1) Segment

AOI ──── (1) Skyline

SnowmeltMap ──── pixel grid
PhenologyResult ──── pixel grid
```

## CSV Schema Specifications

### index.csv (ingest output)
```csv
path
/data/images/MRD_N_NikL_RGB_20240501_0800.jpg
/data/images/MRD_N_NikL_RGB_20240501_0900.jpg
```

### profile.csv
```csv
path,filename,site,azimuth,camera,band,timestamp,width,height,filesize,exif_ok,mean_luma,std_luma,p05_luma,p50_luma,p95_luma,sat_ratio,parse_status
```

### seg/labels.csv
```csv
path,mask_sky_area,mask_cloud_area,mask_fog_area,mask_snow_area,mask_sky_path,mask_cloud_path,mask_fog_path,mask_snow_path
```

### profile_qc.csv
```csv
path,is_usable,reason_codes,brightness,contrast,fog_score,cloud_score,cloud_aoi_ratio
```

### segments.csv
```csv
segment_id,start_ts,end_ts,n_frames,anchor_path,ref_path,status,H_to_global,reproj_err,dx_med,dy_med,peak_med
```

### warp_params.csv
```csv
path,segment_id,dx,dy,rotation_deg,scale,status,method
```

## State Transitions

### Image Processing State
```
[discovered] → ingest → [indexed]
[indexed] → profile → [profiled]
[profiled] → segment → [segmented]
[segmented] → qc → [qc_passed | qc_failed]
[qc_passed] → align → [aligned | align_failed]
```

### Segment State
```
[created] → select_anchor → [anchored]
[anchored] → match → [aligned | failed]
[failed] → retry (if rounds < max) → [aligned | failed]
```

## Validation Rules

1. **Path uniqueness**: No duplicate paths in any CSV
2. **Timestamp ordering**: Images sorted by timestamp within site/camera
3. **Segment continuity**: No gaps in segment_id sequence
4. **Ratio bounds**: All ratio fields in [0, 1]
5. **DOY bounds**: Day of year in [1, 366]
6. **Status codes**: Only predefined values allowed
7. **FK integrity**: All foreign keys reference existing records
