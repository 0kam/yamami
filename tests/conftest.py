"""
Pytest configuration and fixtures for yamami tests.

Uses real image data from test_data_micro for all tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Path to real test data
TEST_DATA_MICRO = Path(__file__).parent.parent / "test_data_micro"
TEST_DATA_FULL = Path(__file__).parent.parent / "test_data_full"


@pytest.fixture
def test_data_dir():
    """
    Return the path to test_data_micro directory.

    Skips tests if the directory doesn't exist.
    """
    if not TEST_DATA_MICRO.exists():
        pytest.skip("test_data_micro directory not found")
    return TEST_DATA_MICRO


@pytest.fixture
def test_images(test_data_dir):
    """
    Return list of image paths from test_data_micro.
    """
    patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.tif", "*.TIF", "*.tiff", "*.TIFF"]
    images = []
    for pattern in patterns:
        images.extend(test_data_dir.glob(pattern))
    return sorted(images)


@pytest.fixture
def sample_image_path(test_images):
    """
    Return a single sample image path for testing.
    """
    if not test_images:
        pytest.skip("No images found in test_data_micro")
    return test_images[0]


@pytest.fixture
def sample_profile(test_data_dir):
    """
    Create a profile DataFrame from test_data_micro images.
    """
    from yamami import ingest

    df = ingest(str(test_data_dir))
    if len(df) == 0:
        pytest.skip("No images found in test_data_micro")

    return df


@pytest.fixture
def output_dir(tmp_path):
    """
    Create a temporary output directory for test results.
    """
    out = tmp_path / "output"
    out.mkdir(exist_ok=True)
    return out
