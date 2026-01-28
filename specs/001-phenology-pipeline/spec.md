# Feature Specification: Yamami Phenology Pipeline

**Feature Branch**: `001-phenology-pipeline`
**Created**: 2026-01-27
**Status**: Draft
**Input**: YAMAMI_PIPELINE_DESIGN.md

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Image Collection and Profiling (Priority: P1)

A researcher has thousands of time-lapse camera images from a fixed mountain observation point. They need to catalog these images with metadata (filename, timestamp, camera type, resolution) and compute basic statistics to identify unusable images (too dark, corrupted) before any analysis begins.

**Why this priority**: This is the foundation of all subsequent analysis. Without properly cataloged and profiled images, no further processing is possible.

**Independent Test**: Can be fully tested with a sample image directory. Delivers a complete profile CSV that identifies all images and their basic quality metrics.

**Acceptance Scenarios**:

1. **Given** a directory containing mixed image formats (JPEG, TIFF, RAW), **When** the researcher runs the ingest command, **Then** the system produces an index listing all JPEG and TIFF files with their paths.
2. **Given** an indexed image collection, **When** the researcher runs the profile command, **Then** the system produces a profile CSV with filename, site, camera, timestamp, dimensions, filesize, and luminance statistics for each image.
3. **Given** a profile CSV, **When** the researcher examines images flagged as "too dark", **Then** those images have luminance values below the configured threshold.

---

### User Story 2 - Quality Control and Image Alignment (Priority: P2)

A researcher needs to identify images where the mountain target is obscured by fog, clouds, or darkness, and detect when the camera angle has shifted (due to wind or SD card replacement). Usable images must be aligned to a reference frame for consistent spatial analysis.

**Why this priority**: Quality control and alignment are prerequisites for accurate phenology analysis. Misaligned or obscured images produce erroneous results.

**Independent Test**: Can be tested with a subset of images containing known fog/cloud conditions and a known camera shift event. Delivers QC flags and alignment parameters.

**Acceptance Scenarios**:

1. **Given** a set of profiled images with SAM3-generated masks, **When** the researcher runs the quality screen command, **Then** images with excessive fog, cloud, or darkness are flagged as unusable with specific reason codes.
2. **Given** a time series with a camera shift event, **When** the researcher runs the alignment command, **Then** the system detects the change point and creates separate alignment segments.
3. **Given** alignment segments and a reference image, **When** alignment is performed, **Then** each segment is transformed to match the reference coordinate system with documented warp parameters.

---

### User Story 3 - Snowmelt Date Estimation (Priority: P3)

A researcher needs to determine when snow melts at each pixel location across the mountain slope throughout the season. This produces a spatial map showing the day-of-year when each location became snow-free.

**Why this priority**: Snowmelt timing is a key phenological indicator and foundation for vegetation greenup analysis.

**Independent Test**: Can be tested with aligned, QC-passed images from a complete snow season. Delivers a snowmelt day-of-year map.

**Acceptance Scenarios**:

1. **Given** aligned images and an AOI mask, **When** the researcher runs the snow extraction command, **Then** each image produces a binary snow/non-snow mask using threshold-based classification.
2. **Given** a time series of snow masks, **When** the researcher runs the snowmelt command, **Then** the system produces a pixel-level map of snowmelt day-of-year.
3. **Given** the snowmelt map, **When** the researcher examines high-reflectance rock areas, **Then** these areas are correctly identified and excluded from snowmelt calculations.

---

### User Story 4 - Vegetation Greenup Analysis (Priority: P4)

A researcher needs to track the seasonal progression of vegetation greenup across the mountain slope by calculating greenness indices and fitting phenological curves to determine green-up and green-down dates.

**Why this priority**: Vegetation phenology is the ultimate research output. Depends on all prior processing stages being complete.

**Independent Test**: Can be tested with a complete growing season of aligned, QC-passed images. Delivers greenness time series and phenology date maps.

**Acceptance Scenarios**:

1. **Given** aligned images and an AOI mask, **When** the researcher runs the GR command, **Then** the system produces daily maximum greenness ratio (GR) values for each pixel.
2. **Given** a seasonal GR time series, **When** the researcher runs the phenology command, **Then** the system fits sigmoid curves and extracts green-up (start) and green-down (end) dates for each pixel.
3. **Given** phenology maps, **When** the researcher runs the visualization command, **Then** the system produces color-coded DOY maps with appropriate legends.

---

### User Story 5 - Results Export and Documentation (Priority: P5)

A researcher needs to export all analysis results in standard formats (CSV, PNG, TIFF) with complete processing logs and parameter documentation for publication and reproducibility.

**Why this priority**: Essential for scientific reproducibility but can be performed after analysis is complete.

**Independent Test**: Can be tested with any completed analysis run. Delivers organized export directory with all outputs and metadata.

**Acceptance Scenarios**:

1. **Given** completed analysis results, **When** the researcher runs the export command, **Then** all outputs are organized in a standardized directory structure.
2. **Given** an export directory, **When** the researcher examines the processing log, **Then** all parameters, software versions, and processing steps are documented.
3. **Given** exported CSV files, **When** loaded by standard tools, **Then** column names and data types are self-documenting and consistent.

---

### Edge Cases

- What happens when an image file is corrupted or unreadable?
  - System logs the error, marks the image as failed in profile, and continues processing remaining images.
- What happens when no reference image is provided for alignment?
  - System prompts user to select a reference image or uses the highest-quality image from the first segment.
- What happens when alignment fails for a segment after maximum retry attempts?
  - System marks the segment as unaligned, excludes it from analysis, and documents the failure.
- What happens when the entire analysis period has cloud cover?
  - System reports insufficient usable images and suggests relaxing QC thresholds or extending the time window.
- What happens when sigmoid fitting fails for phenology?
  - System marks pixels as "fit failed" and provides diagnostic information about the time series.

## Requirements *(mandatory)*

### Functional Requirements

#### Data Ingestion and Profiling
- **FR-001**: System MUST scan a directory and identify all JPEG and TIFF image files
- **FR-002**: System MUST extract metadata from filenames following the pattern `{site}_{azimuth}_{camera}_{band}_{yyyymmdd}_{hhmm}.jpg`
- **FR-003**: System MUST compute luminance statistics (mean, std, percentiles) for each image
- **FR-004**: System MUST read EXIF data when available and record extraction status

#### Semantic Segmentation
- **FR-005**: System MUST segment images using text prompts to identify sky, cloud, fog, and snow regions
- **FR-006**: System MUST support configurable inference resolution with automatic rescaling of masks to original dimensions
- **FR-007**: System MUST output segmentation masks as PNG files with corresponding metadata CSV

#### Quality Control
- **FR-008**: System MUST classify images as usable/unusable based on configurable brightness, fog, and cloud thresholds
- **FR-009**: System MUST produce standardized reason codes for each quality rejection
- **FR-010**: System MUST support both image-level QC (exclude entire image) and pixel-level QC (mask regions)

#### Image Alignment
- **FR-011**: System MUST detect camera shift change points from edge-based phase correlation
- **FR-012**: System MUST segment the time series into stable camera position intervals
- **FR-013**: System MUST select anchor images from each segment based on quality scores
- **FR-014**: System MUST match segments to reference images and compute geometric transformations
- **FR-015**: System MUST apply transformations to all images within each segment
- **FR-016**: System MUST support multiple matching methods (feature-based and learning-based)
- **FR-017**: System MUST record all alignment parameters in reproducible format

#### Skyline and AOI
- **FR-018**: System MUST extract skyline from reference images using blue channel gradient analysis
- **FR-019**: System MUST generate analysis-of-interest (AOI) masks excluding sky regions
- **FR-020**: System MUST support user selection of reference images for skyline extraction

#### Snow Analysis
- **FR-021**: System MUST extract snow regions using threshold-based classification on RGB sum
- **FR-022**: System MUST estimate pixel-level snowmelt day-of-year using multi-stage temporal analysis
- **FR-023**: System MUST handle false positives from high-reflectance rocks

#### Greenness and Phenology
- **FR-024**: System MUST compute greenness ratio (GR = G/(R+G+B)) for each pixel
- **FR-025**: System MUST aggregate to daily maximum GR to reduce noise
- **FR-026**: System MUST fit sigmoid/double-sigmoid curves to GR time series
- **FR-027**: System MUST extract green-up and green-down dates from curve inflection points

#### Visualization and Export
- **FR-028**: System MUST produce DOY-colored maps with legends
- **FR-029**: System MUST export all outputs in CSV (tables) and PNG/TIFF (images) formats
- **FR-030**: System MUST document all processing parameters and software versions in logs

#### Packaging and Distribution
- **FR-031**: System MUST be installable via `pip install yamami`
- **FR-032**: System MUST use modern Python packaging (pyproject.toml, src/ layout)
- **FR-033**: System MUST support Python 3.9-3.12

#### Documentation
- **FR-036**: System MUST provide Sphinx-based documentation hosted on ReadTheDocs
- **FR-037**: System MUST provide documentation in both English and Japanese
- **FR-038**: System MUST include API reference generated from docstrings
- **FR-039**: System MUST include quickstart guide with example workflow
- **FR-040**: System MUST include installation instructions for all dependency options

### Key Entities

- **Image**: A single camera capture with path, timestamp, site, camera identifiers, and computed statistics
- **Profile**: Metadata and quality metrics for an image (luminance stats, parse status, QC flags)
- **Segment**: A contiguous time period with stable camera position (segment_id, start/end timestamps, anchor image, alignment status)
- **Mask**: A binary or categorical image identifying regions (sky, cloud, fog, snow, AOI)
- **Warp Parameters**: Geometric transformation parameters linking an image to the reference coordinate system
- **Snowmelt Map**: Pixel-level day-of-year values indicating when snow disappeared
- **Phenology Map**: Pixel-level day-of-year values for green-up and green-down dates
- **GR Time Series**: Daily greenness ratio values per pixel over the analysis period

## Clarifications

### Session 2026-01-27

- Q: 処理パフォーマンス目標は？ → A: 500画像/時間（フルシーズン約10時間＝一晩）
- Q: SAM3セグメンテーションのGPU要件は？ → A: CUDA対応GPU必須（CPUフォールバックなし）
- Q: 複数サイト・カメラの同時処理は？ → A: 単一サイト・カメラ単位で処理（複数実行はユーザー管理）
- Q: パッケージ構造は？ → A: シンプルなPythonライブラリ（alproj参考、src/yamami/レイアウト）
- Q: 配布方法は？ → A: pip installでインストール可能（pyproject.toml、PyPI公開対応）
- Q: ドキュメントは？ → A: Readthedocs対応、Sphinx + RTDテーマ、日英両言語
- Q: インターフェースは？ → A: Python APIのみ（CLIは不要）
- Q: 設定ファイルは？ → A: config.yaml不要、関数引数でパラメータを渡す
- Q: SAM3のセグメンテーションプロンプトは？ → A: デフォルト ["sky", "cloud", "fog", "snow"]、関数引数で変更可能
- Q: アライメント検出パラメータは？ → A: 関数引数で設定（reproj_err_max, max_rounds等）

## Assumptions

- Images follow a consistent naming convention that encodes site, camera, and timestamp
- Camera position changes are discrete events (not gradual drift) detectable via edge correlation
- One reference image can adequately represent the target coordinate system for alignment
- Snow can be distinguished from non-snow primarily by total RGB intensity
- Vegetation greenup follows a sigmoid-like temporal pattern amenable to curve fitting
- Processing will be performed on a workstation with sufficient memory for batch image operations
- GPU is required: SAM3 segmentation requires a CUDA-capable GPU (no CPU fallback)
- Users have basic familiarity with Python and Jupyter notebooks
- Each pipeline execution processes a single site and camera combination; users manage parallel execution for multiple sites/cameras externally
- Library follows alproj's simple architecture: minimal module count, clear separation of concerns
- Documentation follows ReadTheDocs conventions with sphinx-rtd-theme
- All parameters are passed as function arguments (no config files)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can process a full season (5000+ images) from ingest to phenology maps in a single workflow
- **SC-002**: 95% of images with visible mountain target pass QC and produce valid phenology data
- **SC-003**: Alignment success rate exceeds 90% for segments with clear mountain visibility
- **SC-004**: Snowmelt and phenology dates are reproducible to ±1 day when reprocessed with identical parameters
- **SC-005**: All processing steps are traceable from output back to input images and configuration
- **SC-006**: Researchers can resume processing from any intermediate stage without reprocessing earlier stages
- **SC-007**: Processing logs capture sufficient detail for troubleshooting any failed image or segment
- **SC-008**: System processes at least 500 images per hour on standard workstation hardware (full season completes overnight)
- **SC-009**: Library installs successfully with `pip install yamami` on Python 3.12 with CUDA-capable GPU for segmentation
- **SC-010**: Documentation builds without errors and is accessible on ReadTheDocs in both English and Japanese
- **SC-011**: New users can complete a basic workflow following the quickstart guide within 30 minutes using Python/Jupyter
