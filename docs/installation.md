# Installation

## Requirements

YAMAMI requires Python 3.12 and a CUDA-capable GPU. SAM3 requires
PyTorch 2.7+ and CUDA 12.6+ for inference.

## Basic Installation

Install the latest stable version from PyPI:

```bash
pip install yamami
```

Note: SAM3 is installed from the official GitHub repository, so `git`
must be available in your environment.

## Installation with Extras

YAMAMI provides optional dependency groups for different use cases:

### Development Dependencies

For testing and development:

```bash
pip install yamami[dev]
```

This installs:
- pytest
- pytest-cov

### Documentation Dependencies

For building documentation:

```bash
pip install yamami[docs]
```

This installs:
- sphinx
- sphinx-rtd-theme
- sphinx-intl

### All Dependencies

Install all optional dependencies:

```bash
pip install yamami[dev,docs]
```

## Installation from Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/NIES/yamami.git
cd yamami
pip install -e .
```

For development:

```bash
pip install -e ".[dev,docs]"
```

## Core Dependencies

YAMAMI depends on the following packages (installed automatically):

| Package | Purpose |
|---------|---------|
| numpy | Numerical computing |
| pandas | Data manipulation |
| opencv-python | Image processing |
| scipy | Scientific computing |
| pillow | Image I/O and EXIF |
| tqdm | Progress bars |
| matplotlib | Visualization |
| sam3 | SAM3 segmentation |
| image-matching-models | Feature matching |

## Verifying Installation

Verify YAMAMI is installed correctly:

```python
import yamami
print(yamami.__version__)
```

Run the test suite:

```bash
pytest tests/
```

## Platform Notes

### macOS

SAM3 requires CUDA; CPU-only/MPS-only environments are not supported for
segmentation.

### Linux

For headless servers, ensure a display backend is available for matplotlib:

```python
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
```

YAMAMI automatically uses the Agg backend when imported.

### Windows
SAM3 requires CUDA; CPU-only environments are not supported for segmentation.

## Troubleshooting

### ImportError: No module named 'yamami'

Ensure YAMAMI is installed in the active Python environment:

```bash
pip show yamami
```

### OpenCV Installation Issues

If opencv-python fails to install, try:

```bash
pip install opencv-python-headless
```

### SAM3 Model Access

SAM3 checkpoints are gated; you must request access and authenticate with
Hugging Face before first use.
