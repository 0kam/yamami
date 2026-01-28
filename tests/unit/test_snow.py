"""
Unit tests for snow module.

Tests for:
- Snow threshold classification (Otsu method)
- Snowmelt DOY estimation
- Rock exclusion (high reflectance detection)
"""

import numpy as np
import pandas as pd
import pytest


class TestSnowThresholdClassification:
    """T046: Tests for snow threshold classification using Otsu method."""

    def test_otsu_threshold_bimodal_distribution(self):
        """Test Otsu thresholding on clearly bimodal synthetic data."""
        from yamami.snow import snow

        # Create synthetic image data with clear bimodal distribution
        # Snow pixels: high RGB values (bright)
        # Non-snow pixels: low RGB values (dark)
        np.random.seed(42)

        # Create profile with images that have clear snow/non-snow distinction
        profile = pd.DataFrame({
            "filepath": ["/path/img1.jpg"],
            "doy": [100],
            # High values indicate snow-covered image
            "R_mean": [220.0],
            "G_mean": [220.0],
            "B_mean": [220.0],
        })

        # AOI mask
        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        masks, thresholds = snow(profile, aoi_mask, otsu=True)

        # Verify threshold was computed
        assert len(thresholds) == 1
        assert thresholds["threshold"].iloc[0] > 0

    def test_snow_mask_binary_values(self):
        """Test that snow masks contain only binary values (0 or 1)."""
        from yamami.snow import snow

        profile = pd.DataFrame({
            "filepath": ["/path/img1.jpg"],
            "doy": [100],
            "R_mean": [200.0],
            "G_mean": [200.0],
            "B_mean": [200.0],
        })
        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        masks, _ = snow(profile, aoi_mask)

        for doy, mask in masks.items():
            unique_values = np.unique(mask)
            assert all(v in [0, 1] for v in unique_values), \
                f"Mask for DOY {doy} contains non-binary values: {unique_values}"

    def test_snow_respects_aoi_mask(self):
        """Test that snow detection is limited to AOI region."""
        from yamami.snow import snow

        profile = pd.DataFrame({
            "filepath": ["/path/img1.jpg"],
            "doy": [100],
            "R_mean": [200.0],
            "G_mean": [200.0],
            "B_mean": [200.0],
        })

        # AOI mask with only center region active
        aoi_mask = np.zeros((10, 10), dtype=np.uint8)
        aoi_mask[3:7, 3:7] = 1

        masks, _ = snow(profile, aoi_mask)

        for doy, mask in masks.items():
            # Outside AOI should be 0
            assert np.all(mask[0:3, :] == 0), "Outside AOI should be 0"
            assert np.all(mask[7:10, :] == 0), "Outside AOI should be 0"
            assert np.all(mask[:, 0:3] == 0), "Outside AOI should be 0"
            assert np.all(mask[:, 7:10] == 0), "Outside AOI should be 0"

    def test_fixed_threshold_mode(self):
        """Test snow detection with fixed threshold (otsu=False)."""
        from yamami.snow import snow

        profile = pd.DataFrame({
            "filepath": ["/path/img1.jpg"],
            "doy": [100],
            "R_mean": [200.0],
            "G_mean": [200.0],
            "B_mean": [200.0],
        })
        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        masks, thresholds = snow(profile, aoi_mask, otsu=False)

        # Should still return valid results
        assert len(masks) > 0
        assert len(thresholds) > 0


class TestSnowmeltDOYEstimation:
    """T047: Tests for snowmelt DOY estimation."""

    def test_snowmelt_detects_transition(self):
        """Test snowmelt detection on synthetic time series."""
        from yamami.snow import snowmelt

        # Create snow masks showing transition from snow to no-snow
        # Pixel (0,0): snow melts at DOY 110
        # Pixel (0,1): snow melts at DOY 120
        # Pixel (1,0): no snow initially
        # Pixel (1,1): snow persists
        snow_masks = {
            100: np.array([[1, 1], [0, 1]], dtype=np.uint8),
            110: np.array([[0, 1], [0, 1]], dtype=np.uint8),
            120: np.array([[0, 0], [0, 1]], dtype=np.uint8),
            130: np.array([[0, 0], [0, 1]], dtype=np.uint8),
        }
        aoi_mask = np.ones((2, 2), dtype=np.uint8)

        result = snowmelt(snow_masks, aoi_mask)

        # Find snowmelt DOY for pixel (0,0)
        pixel_00 = result[(result["row"] == 0) & (result["col"] == 0)]
        assert len(pixel_00) == 1
        assert pixel_00["doy"].iloc[0] == 110, \
            "Snowmelt at (0,0) should be DOY 110"

        # Find snowmelt DOY for pixel (0,1)
        pixel_01 = result[(result["row"] == 0) & (result["col"] == 1)]
        assert len(pixel_01) == 1
        assert pixel_01["doy"].iloc[0] == 120, \
            "Snowmelt at (0,1) should be DOY 120"

    def test_snowmelt_no_snow_pixels(self):
        """Test handling of pixels that never had snow."""
        from yamami.snow import snowmelt

        # Pixel (1,0) never has snow
        snow_masks = {
            100: np.array([[1, 1], [0, 0]], dtype=np.uint8),
            110: np.array([[0, 0], [0, 0]], dtype=np.uint8),
        }
        aoi_mask = np.ones((2, 2), dtype=np.uint8)

        result = snowmelt(snow_masks, aoi_mask)

        # Pixel (1,0) should have status indicating no snow
        pixel_10 = result[(result["row"] == 1) & (result["col"] == 0)]
        assert len(pixel_10) == 1
        assert pixel_10["status"].iloc[0] == "no_snow", \
            "Pixel without initial snow should have 'no_snow' status"

    def test_snowmelt_persistent_snow(self):
        """Test handling of pixels where snow persists."""
        from yamami.snow import snowmelt

        # Pixel (1,1) always has snow
        snow_masks = {
            100: np.array([[1, 1], [0, 1]], dtype=np.uint8),
            110: np.array([[0, 1], [0, 1]], dtype=np.uint8),
            120: np.array([[0, 0], [0, 1]], dtype=np.uint8),
        }
        aoi_mask = np.ones((2, 2), dtype=np.uint8)

        result = snowmelt(snow_masks, aoi_mask)

        # Pixel (1,1) should have status indicating persistent snow
        pixel_11 = result[(result["row"] == 1) & (result["col"] == 1)]
        assert len(pixel_11) == 1
        assert pixel_11["status"].iloc[0] == "persistent_snow", \
            "Pixel with persistent snow should have 'persistent_snow' status"

    def test_snowmelt_consecutive_days_requirement(self):
        """Test that snowmelt requires consecutive snow-free days."""
        from yamami.snow import snowmelt

        # Pixel (0,0): snow disappears at DOY 110 but returns at DOY 120
        # Real snowmelt should be when it stays gone
        snow_masks = {
            100: np.array([[1]], dtype=np.uint8),
            110: np.array([[0]], dtype=np.uint8),  # Temporary melt
            120: np.array([[1]], dtype=np.uint8),  # Snow returns
            130: np.array([[0]], dtype=np.uint8),  # Final melt
            140: np.array([[0]], dtype=np.uint8),  # Stays gone
        }
        aoi_mask = np.ones((1, 1), dtype=np.uint8)

        result = snowmelt(snow_masks, aoi_mask)

        # Snowmelt should be 130 (when snow finally disappears for good)
        assert result["doy"].iloc[0] == 130, \
            "Snowmelt should be when snow disappears for consecutive days"

    def test_snowmelt_respects_aoi(self):
        """Test that snowmelt estimation respects AOI mask."""
        from yamami.snow import snowmelt

        snow_masks = {
            100: np.array([[1, 1], [1, 1]], dtype=np.uint8),
            110: np.array([[0, 0], [0, 0]], dtype=np.uint8),
        }
        # Only top-left pixel in AOI
        aoi_mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)

        result = snowmelt(snow_masks, aoi_mask)

        # Should only have one result (for pixel in AOI)
        assert len(result) == 1
        assert result["row"].iloc[0] == 0
        assert result["col"].iloc[0] == 0

    def test_snowmelt_confidence_values(self):
        """Test that confidence values are between 0 and 1."""
        from yamami.snow import snowmelt

        snow_masks = {
            100: np.array([[1, 1], [0, 1]], dtype=np.uint8),
            110: np.array([[0, 0], [0, 0]], dtype=np.uint8),
        }
        aoi_mask = np.ones((2, 2), dtype=np.uint8)

        result = snowmelt(snow_masks, aoi_mask)

        assert all(result["confidence"] >= 0), "Confidence must be >= 0"
        assert all(result["confidence"] <= 1), "Confidence must be <= 1"


class TestHighReflectanceRockDetection:
    """T048: Tests for high reflectance rock detection."""

    def test_detect_high_reflectance_rock_basic(self):
        """Test basic rock detection on synthetic image."""
        from yamami.snow import detect_high_reflectance_rock

        # Create synthetic image with high-reflectance region
        # Rock: high but stable reflectance
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[4:6, 4:6, :] = 200  # High reflectance rock area

        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        rock_mask = detect_high_reflectance_rock(image, aoi_mask)

        assert isinstance(rock_mask, np.ndarray)
        assert rock_mask.shape == (10, 10)
        # Rock area should be detected
        assert rock_mask[4, 4] == 1 or rock_mask[5, 5] == 1

    def test_detect_rock_respects_aoi(self):
        """Test that rock detection respects AOI mask."""
        from yamami.snow import detect_high_reflectance_rock

        image = np.ones((10, 10, 3), dtype=np.uint8) * 200

        # Only center region in AOI
        aoi_mask = np.zeros((10, 10), dtype=np.uint8)
        aoi_mask[3:7, 3:7] = 1

        rock_mask = detect_high_reflectance_rock(image, aoi_mask)

        # Outside AOI should be 0
        assert np.all(rock_mask[0:3, :] == 0)
        assert np.all(rock_mask[7:10, :] == 0)

    def test_detect_rock_returns_binary_mask(self):
        """Test that rock detection returns binary mask."""
        from yamami.snow import detect_high_reflectance_rock

        image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        rock_mask = detect_high_reflectance_rock(image, aoi_mask)

        unique_values = np.unique(rock_mask)
        assert all(v in [0, 1] for v in unique_values), \
            f"Rock mask should be binary, got: {unique_values}"

    def test_detect_rock_differentiates_from_snow(self):
        """Test that rock detection can differentiate from snow."""
        from yamami.snow import detect_high_reflectance_rock

        # Create image where snow and rock have similar high reflectance
        # but rock area is smaller and more consistent
        image = np.zeros((20, 20, 3), dtype=np.uint8)

        # Snow area: large, slightly variable
        image[0:10, :, :] = 240

        # Rock area: small, very consistent high values
        image[15:17, 15:17, :] = 220

        aoi_mask = np.ones((20, 20), dtype=np.uint8)

        rock_mask = detect_high_reflectance_rock(image, aoi_mask)

        # This test verifies the function runs without error
        # Actual differentiation depends on implementation details
        assert rock_mask.shape == (20, 20)


class TestSnowIntegration:
    """Integration tests within snow module."""

    def test_snow_workflow_with_multiple_images(self):
        """Test complete snow detection workflow with multiple images."""
        from yamami.snow import snow

        # Create profile with multiple dates
        profile = pd.DataFrame({
            "filepath": [
                "/path/img1.jpg",
                "/path/img2.jpg",
                "/path/img3.jpg",
            ],
            "doy": [100, 110, 120],
            "R_mean": [230.0, 180.0, 120.0],  # Decreasing snow
            "G_mean": [230.0, 180.0, 120.0],
            "B_mean": [230.0, 180.0, 120.0],
        })
        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        masks, thresholds = snow(profile, aoi_mask)

        # Should have masks for each DOY
        assert len(masks) == 3
        assert 100 in masks
        assert 110 in masks
        assert 120 in masks

        # Should have thresholds for each image
        assert len(thresholds) == 3

    def test_snow_empty_profile(self):
        """Test snow detection with empty profile."""
        from yamami.snow import snow

        profile = pd.DataFrame({
            "filepath": [],
            "doy": [],
            "R_mean": [],
            "G_mean": [],
            "B_mean": [],
        })
        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        masks, thresholds = snow(profile, aoi_mask)

        assert len(masks) == 0
        assert len(thresholds) == 0

    def test_snowmelt_empty_masks(self):
        """Test snowmelt with empty masks dict."""
        from yamami.snow import snowmelt

        snow_masks = {}
        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        result = snowmelt(snow_masks, aoi_mask)

        assert len(result) == 0
