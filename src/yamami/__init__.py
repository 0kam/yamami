"""
YAMAMI: Mountain Phenology Pipeline

A Python library for analyzing time-lapse camera images from fixed mountain
observation points to extract snowmelt timing and vegetation greenup phenology.

YAMAMI provides a complete pipeline for:

- Image ingestion and profiling with metadata extraction
- Semantic segmentation of sky, cloud, fog, and snow using SAM
- Quality control and camera shift detection
- Image alignment to reference coordinate system
- Snowmelt date estimation at pixel level
- Vegetation greenness ratio (GR) calculation
- Phenology curve fitting (double-sigmoid) for greenup/greendown dates
- Visualization and export of results

Example usage:
    >>> import yamami
    >>> profiles = yamami.ingest("/path/to/images")
    >>> # See documentation for full workflow

"""

__version__ = "0.1.0"

from yamami.align import align
from yamami.aoi import aoi
from yamami.export import export
from yamami.ingest import ingest
from yamami.logging import get_logger
from yamami.parser import parse_filename
from yamami.phenology import gr, phenology
from yamami.qc import pre_qc, ridge_qc
from yamami.ridgeline import (
    compute_blue_ratio,
    detect_ridgeline,
    detect_segments,
    get_sam_ridge,
    load_mask,
    match_ridgelines_ransac,
    max_corner_displacement,
)
from yamami.segment import segment
from yamami.snow import snow, snowmelt
from yamami.viz import create_grid_image, draw_ridgeline, viz

# Public API - will be populated as modules are implemented
__all__ = [
    "__version__",
    "align",
    "aoi",
    "compute_blue_ratio",
    "create_grid_image",
    "detect_ridgeline",
    "detect_segments",
    "draw_ridgeline",
    "export",
    "get_logger",
    "get_sam_ridge",
    "gr",
    "ingest",
    "load_mask",
    "match_ridgelines_ransac",
    "max_corner_displacement",
    "parse_filename",
    "phenology",
    "pre_qc",
    "ridge_qc",
    "segment",
    "snow",
    "snowmelt",
    "viz",
]
