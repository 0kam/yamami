"""
Contract tests for snow module.

These tests verify the public API contracts:
- snow() returns tuple (dict of mask arrays, DataFrame with thresholds)
- snowmelt() returns DataFrame with doy column
"""

import numpy as np
import pandas as pd
import pytest


class TestSnowContract:
    """Contract tests for snow() function."""

    def test_snow_returns_tuple(self):
        """T044: snow() returns a tuple of (dict, DataFrame)."""
        from yamami.snow import snow

        # Create minimal synthetic data
        # Profile DataFrame with required columns
        profile = pd.DataFrame({
            "filepath": ["/path/img1.jpg", "/path/img2.jpg"],
            "doy": [100, 110],
            "R_mean": [200.0, 100.0],
            "G_mean": [200.0, 100.0],
            "B_mean": [200.0, 100.0],
        })

        # AOI mask (10x10 for simplicity)
        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        result = snow(profile, aoi_mask)

        # Verify return type
        assert isinstance(result, tuple), "snow() must return a tuple"
        assert len(result) == 2, "snow() must return exactly 2 elements"

        masks, thresholds = result

        # Verify masks is a dict
        assert isinstance(masks, dict), "First element must be a dict of masks"

        # Verify thresholds is a DataFrame
        assert isinstance(thresholds, pd.DataFrame), \
            "Second element must be a DataFrame"

    def test_snow_masks_are_numpy_arrays(self):
        """T044: snow() mask values are numpy arrays."""
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

        for key, mask in masks.items():
            assert isinstance(mask, np.ndarray), \
                f"Mask for {key} must be a numpy array"

    def test_snow_thresholds_dataframe_columns(self):
        """T044: snow() thresholds DataFrame has required columns."""
        from yamami.snow import snow

        profile = pd.DataFrame({
            "filepath": ["/path/img1.jpg"],
            "doy": [100],
            "R_mean": [200.0],
            "G_mean": [200.0],
            "B_mean": [200.0],
        })
        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        _, thresholds = snow(profile, aoi_mask)

        # Verify required columns exist
        required_columns = ["filepath", "threshold"]
        for col in required_columns:
            assert col in thresholds.columns, \
                f"Column '{col}' must be in thresholds DataFrame"


class TestSnowmeltContract:
    """Contract tests for snowmelt() function."""

    def test_snowmelt_returns_dataframe(self):
        """T045: snowmelt() returns a DataFrame."""
        from yamami.snow import snowmelt

        # Create synthetic snow masks dict
        snow_masks = {
            100: np.array([[1, 1], [0, 0]], dtype=np.uint8),
            110: np.array([[1, 0], [0, 0]], dtype=np.uint8),
            120: np.array([[0, 0], [0, 0]], dtype=np.uint8),
        }
        aoi_mask = np.ones((2, 2), dtype=np.uint8)

        result = snowmelt(snow_masks, aoi_mask)

        assert isinstance(result, pd.DataFrame), \
            "snowmelt() must return a DataFrame"

    def test_snowmelt_has_doy_column(self):
        """T045: snowmelt() result has 'doy' column."""
        from yamami.snow import snowmelt

        snow_masks = {
            100: np.array([[1, 1], [0, 0]], dtype=np.uint8),
            110: np.array([[0, 0], [0, 0]], dtype=np.uint8),
        }
        aoi_mask = np.ones((2, 2), dtype=np.uint8)

        result = snowmelt(snow_masks, aoi_mask)

        assert "doy" in result.columns, \
            "snowmelt() result must have 'doy' column"

    def test_snowmelt_has_required_columns(self):
        """T045: snowmelt() result has all required columns."""
        from yamami.snow import snowmelt

        snow_masks = {
            100: np.array([[1, 1], [0, 0]], dtype=np.uint8),
            110: np.array([[0, 0], [0, 0]], dtype=np.uint8),
        }
        aoi_mask = np.ones((2, 2), dtype=np.uint8)

        result = snowmelt(snow_masks, aoi_mask)

        required_columns = ["row", "col", "doy", "confidence", "status"]
        for col in required_columns:
            assert col in result.columns, \
                f"Column '{col}' must be in snowmelt result"
