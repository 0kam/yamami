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
    >>> index = yamami.ingest("/path/to/images")
    >>> profile = yamami.profile(index)
    >>> # See documentation for full workflow

"""

__version__ = "0.1.0"

from yamami.align import align
from yamami.aoi import aoi
from yamami.export import export
from yamami.ingest import ingest, profile
from yamami.logging import get_logger
from yamami.parser import parse_filename
from yamami.phenology import gr, phenology
from yamami.qc import pre_qc, qc
from yamami.segment import segment
from yamami.snow import snow, snowmelt
from yamami.viz import viz

# Public API - will be populated as modules are implemented
__all__ = [
    "__version__",
    "align",
    "aoi",
    "export",
    "get_logger",
    "gr",
    "ingest",
    "parse_filename",
    "phenology",
    "pre_qc",
    "profile",
    "qc",
    "segment",
    "snow",
    "snowmelt",
    "viz",
]
