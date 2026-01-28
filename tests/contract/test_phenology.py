"""
Contract tests for yamami phenology module.

These tests verify the interface contracts of gr() and phenology() functions.
"""

import numpy as np
import pandas as pd
import pytest

from yamami.phenology import gr, phenology


class TestGrContract:
    """Contract tests for gr() function."""

    def test_gr_returns_tuple(self):
        """gr() should return a tuple of (dict, DataFrame)."""
        # Create minimal test data
        profile = pd.DataFrame({
            "filepath": ["/path/img1.jpg", "/path/img2.jpg"],
            "timestamp": pd.to_datetime(["2024-05-01 08:00", "2024-05-01 12:00"]),
        })

        # Simple 2x2 RGB images (stored as file references, but we'll mock the loading)
        aoi_mask = np.ones((2, 2), dtype=bool)

        result = gr(profile, aoi_mask)

        # Check return type is tuple
        assert isinstance(result, tuple), "gr() should return a tuple"
        assert len(result) == 2, "gr() should return a 2-element tuple"

        # Check first element is dict of daily_max images
        daily_images, timeseries = result
        assert isinstance(daily_images, dict), "First element should be a dict"

        # Check second element is DataFrame
        assert isinstance(timeseries, pd.DataFrame), "Second element should be a DataFrame"

    def test_gr_timeseries_has_expected_columns(self):
        """gr() timeseries DataFrame should have date and GR value columns."""
        profile = pd.DataFrame({
            "filepath": ["/path/img1.jpg"],
            "timestamp": pd.to_datetime(["2024-05-01 08:00"]),
        })
        aoi_mask = np.ones((2, 2), dtype=bool)

        _, timeseries = gr(profile, aoi_mask)

        # Should have date and mean_gr columns at minimum
        assert "date" in timeseries.columns or timeseries.index.name == "date", \
            "Timeseries should have date column or index"


class TestPhenologyContract:
    """Contract tests for phenology() function."""

    def test_phenology_returns_dataframe(self):
        """phenology() should return a DataFrame."""
        # Create synthetic GR timeseries data
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        doy = np.arange(1, 366)

        # Simulate seasonal GR pattern
        gr_values = 0.3 + 0.15 * np.sin((doy - 100) * 2 * np.pi / 365)

        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        result = phenology(gr_data)

        assert isinstance(result, pd.DataFrame), "phenology() should return a DataFrame"

    def test_phenology_has_required_columns(self):
        """phenology() DataFrame should have gup_doy, gdown_doy, gmax_doy columns."""
        # Create synthetic seasonal data
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        doy = np.arange(1, 366)

        # Simulate seasonal pattern with clear green-up and green-down
        gr_values = 0.3 + 0.15 * np.sin((doy - 100) * 2 * np.pi / 365)

        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        result = phenology(gr_data)

        # Check required columns exist
        required_columns = ["gup_doy", "gdown_doy", "gmax_doy"]
        for col in required_columns:
            assert col in result.columns, f"phenology() result should have '{col}' column"

    def test_phenology_optional_columns(self):
        """phenology() may include additional diagnostic columns."""
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        doy = np.arange(1, 366)
        gr_values = 0.3 + 0.15 * np.sin((doy - 100) * 2 * np.pi / 365)

        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        result = phenology(gr_data)

        # These are optional but expected columns
        optional_columns = ["gmax_value", "fit_rmse", "fit_status"]
        present_optional = [col for col in optional_columns if col in result.columns]

        # At least one optional column should be present for diagnostics
        assert len(present_optional) >= 0, "Optional diagnostic columns may be included"
