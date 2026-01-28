"""
Unit tests for aoi module.

Tests skyline detection and AOI mask generation.
"""

import numpy as np
import pandas as pd
import pytest


class TestSkylineDetection:
    """Tests for skyline detection."""

    def test_detect_skyline_horizontal_boundary(self):
        """Test skyline detection on image with horizontal sky/ground boundary."""
        from yamami.aoi import _detect_skyline

        # Create image with clear horizontal boundary
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Sky (top half) - high blue
        img[:50, :, 0] = 200  # Blue
        img[:50, :, 1] = 150
        img[:50, :, 2] = 100
        # Ground (bottom half) - low blue
        img[50:, :, 0] = 50
        img[50:, :, 1] = 100
        img[50:, :, 2] = 50

        skyline = _detect_skyline(img)

        # Skyline should be around y=50
        assert isinstance(skyline, pd.DataFrame)
        assert "x" in skyline.columns
        assert "y" in skyline.columns
        # Check skyline y values are around 50
        assert 40 <= skyline["y"].mean() <= 60

    def test_detect_skyline_returns_full_width(self):
        """Test skyline covers full image width."""
        from yamami.aoi import _detect_skyline

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:50, :, 0] = 200  # Sky

        skyline = _detect_skyline(img)

        # Should have points across full width
        assert skyline["x"].min() == 0
        assert skyline["x"].max() == 199


class TestAOIMaskGeneration:
    """Tests for AOI mask generation."""

    def test_generate_aoi_mask_shape(self):
        """Test AOI mask has correct shape."""
        from yamami.aoi import _generate_aoi_mask

        img_shape = (100, 200)
        skyline = pd.DataFrame({"x": list(range(200)), "y": [50] * 200})

        mask = _generate_aoi_mask(skyline, img_shape)

        assert mask.shape == img_shape

    def test_generate_aoi_mask_binary(self):
        """Test AOI mask is binary."""
        from yamami.aoi import _generate_aoi_mask

        img_shape = (100, 100)
        skyline = pd.DataFrame({"x": list(range(100)), "y": [50] * 100})

        mask = _generate_aoi_mask(skyline, img_shape)

        unique_values = np.unique(mask)
        assert all(v in [0, 255] for v in unique_values)

    def test_generate_aoi_mask_below_skyline(self):
        """Test AOI mask is below skyline (ground region)."""
        from yamami.aoi import _generate_aoi_mask

        img_shape = (100, 100)
        skyline = pd.DataFrame({"x": list(range(100)), "y": [50] * 100})

        mask = _generate_aoi_mask(skyline, img_shape)

        # Above skyline should be 0 (sky)
        assert np.all(mask[:45, :] == 0)
        # Below skyline should be 255 (AOI)
        assert np.all(mask[55:, :] == 255)

    def test_generate_aoi_mask_irregular_skyline(self):
        """Test AOI mask handles irregular skyline."""
        from yamami.aoi import _generate_aoi_mask

        img_shape = (100, 100)
        # Irregular skyline (mountain-like)
        skyline = pd.DataFrame(
            {
                "x": list(range(100)),
                "y": [30 + int(20 * np.sin(x * np.pi / 50)) for x in range(100)],
            }
        )

        mask = _generate_aoi_mask(skyline, img_shape)

        # Should still be binary
        unique_values = np.unique(mask)
        assert all(v in [0, 255] for v in unique_values)


class TestAOIFunction:
    """Tests for main aoi() function."""

    def test_aoi_creates_output_dir(self, sample_ref_image, tmp_path):
        """Test aoi creates output directory."""
        from yamami.aoi import aoi
        import os

        output_dir = str(tmp_path / "aoi_output")
        _, _ = aoi(sample_ref_image, output_dir=output_dir)

        assert os.path.exists(output_dir)

    def test_aoi_saves_skyline(self, sample_ref_image, tmp_path):
        """Test aoi saves skyline to file."""
        from yamami.aoi import aoi
        import os

        output_dir = str(tmp_path / "aoi_output")
        _, _ = aoi(sample_ref_image, output_dir=output_dir)

        skyline_path = os.path.join(output_dir, "skyline.csv")
        assert os.path.exists(skyline_path)

    def test_aoi_saves_mask(self, sample_ref_image, tmp_path):
        """Test aoi saves mask to file."""
        from yamami.aoi import aoi
        import os

        output_dir = str(tmp_path / "aoi_output")
        _, _ = aoi(sample_ref_image, output_dir=output_dir)

        mask_path = os.path.join(output_dir, "aoi_mask.png")
        assert os.path.exists(mask_path)

    def test_aoi_returns_correct_types(self, sample_ref_image, tmp_path):
        """Test aoi returns correct types."""
        from yamami.aoi import aoi

        output_dir = str(tmp_path / "aoi_output")
        skyline, aoi_mask = aoi(sample_ref_image, output_dir=output_dir)

        assert isinstance(skyline, pd.DataFrame)
        assert isinstance(aoi_mask, np.ndarray)

    def test_aoi_mask_matches_image_size(self, sample_ref_image, tmp_path):
        """Test AOI mask matches reference image size."""
        from yamami.aoi import aoi
        import cv2

        output_dir = str(tmp_path / "aoi_output")
        _, aoi_mask = aoi(sample_ref_image, output_dir=output_dir)

        # Load reference image to check size
        ref_img = cv2.imread(sample_ref_image)

        assert aoi_mask.shape[:2] == ref_img.shape[:2]


# Fixtures
@pytest.fixture
def sample_ref_image(tmp_path):
    """Create a sample reference image for testing."""
    import cv2

    img_path = tmp_path / "ref_image.jpg"

    # Create image with sky-like gradient (blue at top, green at bottom)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Sky region (top half) - blue
    img[:50, :, 0] = 200  # Blue channel
    img[:50, :, 1] = 150
    img[:50, :, 2] = 100

    # Ground region (bottom half) - green
    img[50:, :, 0] = 50
    img[50:, :, 1] = 150
    img[50:, :, 2] = 50

    cv2.imwrite(str(img_path), img)
    return str(img_path)
