YAMAMI Documentation
====================

**YAMAMI**: Mountain Phenology Pipeline

A Python library for analyzing time-lapse camera images from fixed mountain
observation points. YAMAMI provides a complete pipeline for processing
seasonal image sequences to extract snowmelt timing and vegetation greenup
phenology across mountain landscapes.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index


Features
--------

- **Image Ingestion**: Scan and profile image collections with metadata extraction
- **Semantic Segmentation**: Segment sky, cloud, fog, and snow using SAM3
- **Quality Control**: Pre- and post-alignment QC filtering
- **Image Alignment**: Camera shift detection and homography-based alignment
- **AOI Extraction**: Automatic skyline detection and area of interest masking
- **Snow Analysis**: Snow detection and pixel-level snowmelt date estimation
- **Phenology Extraction**: Greenness ratio calculation and double-sigmoid fitting
- **Visualization**: Color-coded DOY maps with publication-quality figures
- **Export**: Structured result export with provenance tracking


Quick Example
-------------

.. code-block:: python

   import yamami

   # Ingest images from directory
   index = yamami.ingest("/path/to/images")
   profile = yamami.profile(index)

   # Segment and filter
   labels, masks = yamami.segment(profile)
   qc_result = yamami.pre_qc(profile, labels)

   # Align images
   segments, warp_params, aligned_dir = yamami.align(
       profile[qc_result["is_usable"]],
       ref="reference.jpg"
   )

   # Extract phenology
   skyline, aoi_mask = yamami.aoi("reference.jpg")
   images, timeseries = yamami.gr(profile, aoi_mask)
   pheno = yamami.phenology(timeseries)

   # Visualize and export
   viz_result = yamami.viz(pheno, output_dir="output/viz")
   export_path = yamami.export("output")


Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install yamami

For development and documentation dependencies, see :doc:`installation`.


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
