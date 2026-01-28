"""
Pytest configuration and fixtures for yamami tests.
"""

import numpy as np
import pytest


@pytest.fixture
def sample_image():
    """
    Create a synthetic test image for testing.

    Returns a 100x100 RGB image with a simple pattern:
    - White background
    - Black square in the center (simulating an object)

    Returns:
        np.ndarray: Test image with shape (100, 100, 3) and dtype uint8.
    """
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White background

    # Add a black square in the center (30x30)
    img[35:65, 35:65, :] = 0

    return img


@pytest.fixture
def sample_grayscale_image():
    """
    Create a synthetic grayscale test image.

    Returns a 100x100 grayscale image with a gradient pattern.

    Returns:
        np.ndarray: Test image with shape (100, 100) and dtype uint8.
    """
    img = np.zeros((100, 100), dtype=np.uint8)

    # Create horizontal gradient
    for i in range(100):
        img[:, i] = int(i * 255 / 99)

    return img


@pytest.fixture
def sample_image_with_markers(sample_image):
    """
    Create a test image with simulated ArUco-like markers.

    Adds four corner markers to the sample image for testing
    marker detection functionality.

    Args:
        sample_image: Base test image from sample_image fixture.

    Returns:
        np.ndarray: Test image with marker patterns in corners.
    """
    img = sample_image.copy()

    # Add simple marker patterns in corners (10x10 black squares)
    marker_size = 10

    # Top-left
    img[5:5 + marker_size, 5:5 + marker_size, :] = 0

    # Top-right
    img[5:5 + marker_size, 85:85 + marker_size, :] = 0

    # Bottom-left
    img[85:85 + marker_size, 5:5 + marker_size, :] = 0

    # Bottom-right
    img[85:85 + marker_size, 85:85 + marker_size, :] = 0

    return img


@pytest.fixture
def test_data_dir(tmp_path):
    """
    Create a temporary directory structure for test data.

    Creates subdirectories for images and outputs within tmp_path.

    Args:
        tmp_path: Pytest's built-in tmp_path fixture.

    Returns:
        dict: Dictionary containing paths to test directories.
    """
    dirs = {
        "root": tmp_path,
        "images": tmp_path / "images",
        "outputs": tmp_path / "outputs",
        "masks": tmp_path / "masks",
    }

    for path in dirs.values():
        path.mkdir(exist_ok=True)

    return dirs


@pytest.fixture
def sample_mask():
    """
    Create a sample binary mask for testing.

    Returns a 100x100 binary mask with a circular region.

    Returns:
        np.ndarray: Binary mask with shape (100, 100) and dtype uint8.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)

    # Create circular mask in center
    center = (50, 50)
    radius = 20

    y, x = np.ogrid[:100, :100]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask[dist_from_center <= radius] = 255

    return mask


@pytest.fixture(autouse=True)
def mock_segment_models(request, monkeypatch):
    """
    Mock SAM3 segmentation for all tests except real_data.

    This keeps unit/integration tests fast and independent of large model files.
    """
    if request.node.get_closest_marker("real_data"):
        return

    import importlib

    seg = importlib.import_module("yamami.segment")

    def fake_load_models(device):
        return (None, None)

    def fake_segment_single_image(img, prompts, model_bundle, resize, device):
        return {p: np.zeros(img.shape[:2], dtype=np.uint8) for p in prompts}

    monkeypatch.setattr(seg, "_load_models", fake_load_models)
    monkeypatch.setattr(seg, "_segment_single_image", fake_segment_single_image)


@pytest.fixture(autouse=True)
def mock_align_matching(request, monkeypatch):
    """
    Mock alignment matching for non-real_data tests to avoid heavy IMM models.
    """
    if request.node.get_closest_marker("real_data"):
        return

    import importlib
    import numpy as np

    align = importlib.import_module("yamami.align")

    def fake_match_images(path1, path2, method="akaze", device="cpu", resize=None):
        # Four non-collinear points
        pts1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
        pts2 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
        return pts1, pts2

    monkeypatch.setattr(align, "_match_images", fake_match_images)
