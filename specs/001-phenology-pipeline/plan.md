# Implementation Plan: Yamami Phenology Pipeline

**Branch**: `001-phenology-pipeline` | **Date**: 2026-01-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-phenology-pipeline/spec.md`

## Summary

Yamami is a Python library for processing time-lapse mountain camera images to extract snowmelt timing and vegetation phenology. The pipeline processes thousands of images through: ingest → profile → segmentation → QC → alignment → skyline/AOI → snow → snowmelt → GR → phenology → visualization → export. Following alproj's architecture, the library is designed as a minimal, pip-installable package with a simple Python API (no CLI).

## Technical Context

**Language/Version**: Python 3.9-3.12
**Primary Dependencies**: numpy, pandas, opencv-python, scipy, pillow, tqdm
**ML Dependencies**: segment-anything (SAM3), image-matching-models (imm)
**Storage**: File-based (CSV for tables, PNG/TIFF for images)
**Testing**: pytest with pytest-cov (TDD mandated by Constitution)
**Target Platform**: Linux, macOS, Windows (workstation with optional GPU)
**Project Type**: Single Python library (API only, no CLI)
**Performance Goals**: 500 images/hour on standard workstation (full season overnight)
**Constraints**: GPU optional (CUDA acceleration when available, CPU fallback)
**Scale/Scope**: 5000+ images per season, single site/camera per run

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Verify against yamami Constitution principles:

- [x] **I. Reproducibility First**: All parameters as function arguments, processing logs with file hashes and software versions, intermediate outputs preserved as CSV/PNG/TIFF
- [x] **II. Pipeline Modularity**: Each stage is independent with clear I/O contracts (DataFrames, PNG masks), each module exposes clear public API
- [x] **III. Simplicity Over Cleverness**: Following alproj's minimal architecture, ~6-8 core modules, standard library preferred
- [x] **IV. Test-Driven Development**: pytest with TDD workflow, contract tests for stage interfaces, integration tests for pipeline flows, coverage ≥80% core / ≥60% utilities
- [x] **V. Traceability and Logging**: Standardized status codes (align_status, qc_status), metadata in all outputs, structured logging with context

## Project Structure

### Documentation (this feature)

```text
specs/001-phenology-pipeline/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/
└── yamami/
    ├── __init__.py          # Package exports and public API
    ├── parser.py            # FR-002: Filename parsing utilities
    ├── logging.py           # FR-030: Structured logging setup
    ├── ingest.py            # FR-001,003,004: Image discovery and profiling
    ├── segment.py           # FR-005~007: SAM3 semantic segmentation
    ├── qc.py                # FR-008~010: Quality control
    ├── align.py             # FR-011~017: Camera shift detection and alignment
    ├── aoi.py               # FR-018~020: Skyline extraction and AOI masks
    ├── snow.py              # FR-021~023: Snow extraction and snowmelt
    ├── phenology.py         # FR-024~027: GR calculation and sigmoid fitting
    ├── viz.py               # FR-028: DOY visualization
    └── export.py            # FR-029~030: Results export and logging

tests/
├── conftest.py              # Shared fixtures
├── contract/                # Inter-stage interface tests
│   ├── test_ingest_profile.py
│   ├── test_segment_qc.py
│   └── test_align_aoi.py
├── integration/             # End-to-end pipeline tests
│   └── test_pipeline.py
└── unit/                    # Module unit tests
    ├── test_ingest.py
    ├── test_segment.py
    ├── test_qc.py
    ├── test_align.py
    ├── test_aoi.py
    ├── test_snow.py
    ├── test_phenology.py
    ├── test_viz.py
    └── test_export.py

docs/
├── conf.py                  # Sphinx configuration
├── index.rst                # Documentation index
├── installation.md          # FR-040: Installation guide
├── quickstart.md            # FR-039: Quickstart guide
├── api/                     # FR-038: API reference (autodoc)
├── locale/                  # FR-037: Japanese translations
│   └── ja/
└── requirements.txt         # Documentation dependencies

pyproject.toml               # FR-031~033: Package configuration
.readthedocs.yaml            # FR-036: ReadTheDocs configuration
README.md                    # Project overview
```

**Structure Decision**: Single Python library following alproj's src/ layout pattern. Each pipeline stage maps to one module for clear separation of concerns per Constitution Principle II.

## Complexity Tracking

No Constitution violations requiring justification. Design follows all five principles.
