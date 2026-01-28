# Research: Yamami Phenology Pipeline

**Date**: 2026-01-27
**Branch**: 001-phenology-pipeline

## Technology Decisions

### 1. Image Processing Stack

**Decision**: OpenCV + NumPy + Pillow

**Rationale**:
- OpenCV provides robust image I/O, geometric transformations, and feature detection
- NumPy enables efficient array operations for pixel-level statistics
- Pillow handles EXIF extraction reliably across formats
- All are well-maintained, cross-platform, and widely used in scientific imaging

**Alternatives considered**:
- scikit-image: Good API but slower for large batch processing
- imageio: Limited geometric transformation support
- rawpy: Only needed for RAW files, which are explicitly out of scope

### 2. Semantic Segmentation (SAM3)

**Decision**: Use segment-anything library with text prompts

**Rationale**:
- SAM3 provides zero-shot segmentation without domain-specific training
- Text prompts ("sky", "cloud", "fog", "snow") are intuitive and maintainable
- Supports configurable inference resolution for performance tuning
- CPU fallback available when GPU not present

**Alternatives considered**:
- Custom trained model: Requires labeled data, maintenance burden
- Traditional thresholding: Less accurate for complex scenes
- DeepLabV3: Requires fine-tuning for mountain landscapes

### 3. Image Alignment

**Decision**: Phase correlation for change point detection + image-matching-models (imm) for precise alignment

**Rationale**:
- Phase correlation is fast and sufficient for detecting camera shifts
- imm provides multiple matching methods (SIFT, LightGlue, LoFTR, RoMa) under unified API
- Follows existing alproj/gcp.py patterns for consistency
- RANSAC/USAC_MAGSAC filtering for robust outlier removal

**Alternatives considered**:
- OpenCV ORB only: Less accurate for mountain textures
- Manual GCP selection: Not scalable for thousands of images
- Dense optical flow: Too computationally expensive

### 4. Sigmoid Fitting for Phenology

**Decision**: scipy.optimize.curve_fit with double-sigmoid model

**Rationale**:
- Double-sigmoid captures both greenup and senescence in single fit
- scipy.optimize is standard and well-documented
- Inflection point extraction is straightforward from fitted parameters
- Can handle pixel-wise fitting with vectorized operations

**Alternatives considered**:
- Savitzky-Golay smoothing: No parametric model for phenology dates
- TIMESAT: External dependency, licensing concerns
- Neural network: Overkill for well-understood sigmoid pattern

### 5. API Design

**Decision**: Simple Python functions with typed arguments

**Rationale**:
- No configuration files needed - all parameters as function arguments
- Matches Jupyter/notebook workflow
- Easy to inspect and document with docstrings
- Type hints provide IDE support and documentation

**Alternatives considered**:
- YAML config files: Adds complexity, not needed for API-only library
- CLI with argparse: Not needed - users call Python functions directly
- Class-based API: Over-engineering for stateless pipeline stages

### 6. Logging

**Decision**: Python logging module with structured formatters

**Rationale**:
- Standard library, no dependencies
- Configurable log levels and handlers
- JSON formatter available for machine parsing
- Constitution Principle V requires detailed traceability

**Alternatives considered**:
- loguru: Nice API but unnecessary dependency
- structlog: Overhead for this use case

### 7. Testing Strategy

**Decision**: pytest + pytest-cov with fixture-based test data

**Rationale**:
- pytest is standard for Python scientific projects
- Fixtures enable shared test data management
- Coverage reporting integrates with CI
- Constitution Principle IV mandates TDD

**Test data strategy**:
- Small synthetic images for unit tests (generated in fixtures)
- Real sample images (5-10) for integration tests (versioned in test_data/)
- Contract tests verify inter-stage CSV/PNG format compatibility

### 8. Documentation

**Decision**: Sphinx with sphinx-rtd-theme, gettext for i18n

**Rationale**:
- ReadTheDocs native integration
- autodoc generates API reference from docstrings
- gettext provides standard i18n workflow for Japanese translation
- Follows alproj documentation structure

**Alternatives considered**:
- MkDocs: Less autodoc support
- pdoc: Limited customization

## Performance Research

### Image Processing Throughput

**Target**: 500 images/hour = ~8.3 images/minute = ~7.2 seconds/image

**Bottleneck analysis**:
1. Image I/O: ~0.5s/image (JPEG decode)
2. SAM3 inference: ~3-5s/image (GPU) or ~30s/image (CPU)
3. Alignment matching: ~1-2s/image (with imm)
4. Statistics computation: ~0.1s/image

**Strategy**:
- Batch processing with progress bars (tqdm)
- SAM3 at reduced resolution (512x512) with mask upscaling
- Cache masks to avoid recomputation
- Parallel I/O where beneficial (but avoid overcomplicating per Principle III)

### Memory Management

**Target**: Process 5000+ images without memory exhaustion

**Strategy**:
- Process images one at a time, write results immediately
- Store only paths in profile CSV, not image data
- Use memory-mapped arrays for large rasters if needed
- Clear intermediate arrays explicitly

## Integration Points

### Input Format

**Image files**: JPEG, TIFF (RAW excluded per spec)
**Filename pattern**: `{site}_{azimuth}_{camera}_{band}_{yyyymmdd}_{hhmm}.jpg`
**Configuration**: YAML file per site/camera

### Output Format

**Tables**: CSV with standardized columns
**Images**: PNG (masks, visualizations), TIFF (georeferenced if needed)
**Logs**: JSON-formatted log files per run

### Inter-stage Contracts

| Stage | Input | Output |
|-------|-------|--------|
| ingest | Image directory | index.csv |
| profile | index.csv | profile.csv |
| segment | profile.csv | seg/labels.csv, seg/masks/*.png |
| pre_qc | profile.csv + masks | profile_pre.csv |
| align | profile_pre.csv + ref image | segments.csv, warp_params.csv, aligned/ |
| aoi | Reference image | skyline.csv, aoi_mask.png |
| qc | profile.csv + masks + AOI | profile_qc.csv |
| snow | Aligned images + AOI | snow/extract/*.png, thresholds.csv |
| snowmelt | snow/extract/ | snowmelt/doy_map.png, doy_map.csv |
| gr | Aligned images + AOI | gr/daily_max/*.png, timeseries.csv |
| phenology | GR time series | phenology/*.png, phenology.csv |
| viz | All maps | viz/*.png |
| export | All outputs | export/ (organized) |

## Open Questions Resolved

All technical decisions made. No NEEDS CLARIFICATION items remain.
