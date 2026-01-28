API Reference
=============

This section contains the complete API documentation for YAMAMI.

.. toctree::
   :maxdepth: 2

   parser
   ingest
   segment
   qc
   align
   aoi
   snow
   phenology
   viz
   export
   logging


Module Overview
---------------

YAMAMI provides the following main modules:

**Data Ingestion**

- :mod:`yamami.parser` - Filename parsing and metadata extraction
- :mod:`yamami.ingest` - Image scanning and profiling

**Image Processing**

- :mod:`yamami.segment` - Semantic segmentation with SAM3
- :mod:`yamami.qc` - Quality control and filtering
- :mod:`yamami.align` - Camera shift detection and alignment
- :mod:`yamami.aoi` - Area of interest extraction

**Analysis**

- :mod:`yamami.snow` - Snow detection and snowmelt estimation
- :mod:`yamami.phenology` - Greenness ratio and phenology extraction

**Output**

- :mod:`yamami.viz` - Visualization generation
- :mod:`yamami.export` - Result export and provenance

**Utilities**

- :mod:`yamami.logging` - Logging configuration


Public API
----------

The following functions are available directly from the ``yamami`` namespace:

.. code-block:: python

   import yamami

   # Ingestion
   yamami.ingest()      # Scan directory for images
   yamami.profile()     # Profile images with metadata and stats

   # Parsing
   yamami.parse_filename()  # Parse yamami filename convention

   # Segmentation and QC
   yamami.segment()     # Semantic segmentation
   yamami.pre_qc()      # Pre-alignment quality control
   yamami.qc()          # Full quality control with AOI

   # Alignment and AOI
   yamami.align()       # Detect shifts and align images
   yamami.aoi()         # Extract area of interest

   # Analysis
   yamami.snow()        # Snow detection
   yamami.snowmelt()    # Snowmelt date estimation
   yamami.gr()          # Greenness ratio calculation
   yamami.phenology()   # Phenology extraction

   # Output
   yamami.viz()         # Generate visualizations
   yamami.export()      # Export results

   # Utilities
   yamami.get_logger()  # Get logger instance
