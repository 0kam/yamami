# yamami Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-02-09

## Active Technologies

- Python 3.12 + numpy, pandas, opencv-python, scipy, pillow, tqdm, matplotlib, torch, sam3, image-matching-models (001-phenology-pipeline)

## Project Structure

```text
src/yamami/
  __init__.py    # Public API exports
  ingest.py      # Image discovery + profiling
  parser.py      # Filename timestamp parsing
  segment.py     # SAM3 semantic segmentation
  qc.py          # pre_qc + ridge_qc
  ridgeline.py   # Ridgeline detection/matching
  align.py       # Round-based alignment + lens distortion
  aoi.py         # AOI extraction
  phenology.py   # GR calculation + phenology fitting
  snow.py        # Snow detection + snowmelt estimation
  viz.py         # Visualization
  export.py      # Result export
  logging.py     # Structured logging
tests/
  conftest.py
  integration/test_real_data.py
```

## Commands

```bash
# 仮想環境を使用
source .venv/bin/activate

# テスト実行
.venv/bin/pytest tests/ -v

# リンター
.venv/bin/ruff check .
```

## Code Style

Python 3.12: Follow standard conventions

## Recent Changes

- 001-phenology-pipeline: Full pipeline implementation (ingest -> segment -> ridge_qc -> align -> aoi -> gr/phenology -> snow/snowmelt -> viz -> export)

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
