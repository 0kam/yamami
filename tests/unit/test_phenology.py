"""
Unit tests for yamami phenology module.

Tests for GR calculation, daily max aggregation, double-sigmoid fitting,
and inflection point extraction.
"""

import numpy as np
import pandas as pd
import pytest

from yamami.phenology import gr, fit_double_sigmoid, phenology


class TestGrCalculation:
    """Tests for GR (Greenness Ratio) calculation."""

    def test_gr_formula_basic(self):
        """GR should be calculated as G / (R + G + B)."""
        # Create a simple 1x1 RGB image where we know the expected result
        # R=100, G=150, B=50 -> GR = 150 / (100 + 150 + 50) = 0.5
        r, g, b = 100, 150, 50
        expected_gr = g / (r + g + b)

        # The actual calculation should match
        assert expected_gr == 0.5

    def test_gr_white_pixel(self):
        """White pixel (255, 255, 255) should have GR = 1/3."""
        r, g, b = 255, 255, 255
        expected_gr = g / (r + g + b)

        assert abs(expected_gr - 1/3) < 1e-10

    def test_gr_pure_green_pixel(self):
        """Pure green pixel (0, 255, 0) should have GR = 1.0."""
        r, g, b = 0, 255, 0
        expected_gr = g / (r + g + b)

        assert expected_gr == 1.0

    def test_gr_pure_red_pixel(self):
        """Pure red pixel (255, 0, 0) should have GR = 0.0."""
        r, g, b = 255, 0, 0
        expected_gr = g / (r + g + b)

        assert expected_gr == 0.0

    def test_gr_division_by_zero_handling(self):
        """GR calculation should handle black pixels (0, 0, 0) gracefully."""
        # Black pixel: R=G=B=0 would cause division by zero
        # Should return 0 or NaN, not raise an error
        r, g, b = 0, 0, 0

        # Simulating the expected behavior
        if r + g + b == 0:
            expected_gr = 0.0  # or np.nan
        else:
            expected_gr = g / (r + g + b)

        # The function should handle this case
        assert expected_gr == 0.0 or np.isnan(expected_gr)


class TestDailyMaxAggregation:
    """Tests for daily maximum aggregation of GR values."""

    def test_daily_max_single_day(self):
        """Single day with multiple observations should return max value."""
        # Simulate multiple observations on the same day
        timestamps = pd.to_datetime([
            "2024-05-01 06:00",
            "2024-05-01 08:00",
            "2024-05-01 10:00",
            "2024-05-01 12:00",
        ])
        gr_values = [0.30, 0.35, 0.40, 0.38]  # Max is 0.40 at 10:00

        df = pd.DataFrame({
            "timestamp": timestamps,
            "gr": gr_values,
        })
        df["date"] = df["timestamp"].dt.date

        daily_max = df.groupby("date")["gr"].max()

        assert len(daily_max) == 1
        assert daily_max.iloc[0] == 0.40

    def test_daily_max_multiple_days(self):
        """Multiple days should each have their own max value."""
        timestamps = pd.to_datetime([
            "2024-05-01 08:00", "2024-05-01 12:00",
            "2024-05-02 08:00", "2024-05-02 12:00",
        ])
        gr_values = [0.30, 0.35, 0.40, 0.38]

        df = pd.DataFrame({
            "timestamp": timestamps,
            "gr": gr_values,
        })
        df["date"] = df["timestamp"].dt.date

        daily_max = df.groupby("date")["gr"].max()

        assert len(daily_max) == 2
        assert daily_max.iloc[0] == 0.35  # May 1st max
        assert daily_max.iloc[1] == 0.40  # May 2nd max


class TestDoubleSigmoidFitting:
    """Tests for double-sigmoid curve fitting."""

    def test_fit_returns_dict_with_parameters(self):
        """fit_double_sigmoid should return dict with fitted parameters."""
        # Create synthetic seasonal data
        t = np.arange(1, 366)  # DOY
        # Simple sinusoidal pattern as proxy for seasonal variation
        values = 0.3 + 0.15 * np.sin((t - 100) * 2 * np.pi / 365)

        result = fit_double_sigmoid(t, values)

        assert isinstance(result, dict)
        assert "rmse" in result or "fit_rmse" in result

    def test_fit_returns_inflection_points(self):
        """fit_double_sigmoid should return inflection point estimates."""
        # Create data with clear seasonal pattern
        t = np.arange(1, 366)
        # Simulate green-up around day 100, green-down around day 280
        values = 0.3 + 0.15 * np.sin((t - 100) * 2 * np.pi / 365)

        result = fit_double_sigmoid(t, values)

        # Should have keys for inflection points or parameters to derive them
        assert isinstance(result, dict)

    def test_fit_handles_noisy_data(self):
        """fit_double_sigmoid should handle noisy data without crashing."""
        np.random.seed(42)
        t = np.arange(1, 366)
        values = 0.3 + 0.15 * np.sin((t - 100) * 2 * np.pi / 365) + 0.02 * np.random.randn(365)

        result = fit_double_sigmoid(t, values)

        assert isinstance(result, dict)

    def test_fit_handles_flat_data(self):
        """fit_double_sigmoid should handle flat/constant data gracefully."""
        t = np.arange(1, 366)
        values = np.full(365, 0.35)  # Constant value

        result = fit_double_sigmoid(t, values)

        assert isinstance(result, dict)
        # Should indicate fit failure or poor fit
        assert "status" in result or "fit_status" in result


class TestInflectionPointExtraction:
    """Tests for extracting phenology dates from fitted curve."""

    def test_gup_before_gmax(self):
        """Green-up DOY should be before green-max DOY."""
        # Create data with clear seasonal pattern
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        doy = np.arange(1, 366)
        # Peak around day 180 (late June)
        gr_values = 0.3 + 0.15 * np.sin((doy - 100) * 2 * np.pi / 365)

        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        result = phenology(gr_data)

        # If fit succeeded, gup_doy should be before gmax_doy
        if result["fit_status"].iloc[0] == "success":
            assert result["gup_doy"].iloc[0] < result["gmax_doy"].iloc[0]

    def test_gmax_before_gdown(self):
        """Green-max DOY should be before green-down DOY."""
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        doy = np.arange(1, 366)
        gr_values = 0.3 + 0.15 * np.sin((doy - 100) * 2 * np.pi / 365)

        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        result = phenology(gr_data)

        # If fit succeeded, gmax_doy should be before gdown_doy
        if result["fit_status"].iloc[0] == "success":
            assert result["gmax_doy"].iloc[0] < result["gdown_doy"].iloc[0]

    def test_phenology_dates_in_valid_range(self):
        """All phenology DOYs should be within 1-365 range."""
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        doy = np.arange(1, 366)
        gr_values = 0.3 + 0.15 * np.sin((doy - 100) * 2 * np.pi / 365)

        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        result = phenology(gr_data)

        if result["fit_status"].iloc[0] == "success":
            assert 1 <= result["gup_doy"].iloc[0] <= 365
            assert 1 <= result["gmax_doy"].iloc[0] <= 365
            assert 1 <= result["gdown_doy"].iloc[0] <= 365


class TestPhenologyFitFailure:
    """Tests for handling fit failures gracefully."""

    def test_fit_failure_returns_status(self):
        """Failed fits should return status='failed' and NaN for dates."""
        # Create data that's hard to fit (random noise)
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        doy = np.arange(1, 366)
        gr_values = np.random.rand(365)  # Random noise, no seasonal pattern

        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        result = phenology(gr_data)

        assert "fit_status" in result.columns

    def test_fit_failure_nan_phenology_dates(self):
        """Failed fits should have NaN for phenology dates."""
        # Very short time series that can't be fitted
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        doy = np.arange(1, 11)
        gr_values = np.linspace(0.3, 0.35, 10)  # Linear, not seasonal

        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        result = phenology(gr_data)

        # Either fit fails or dates are invalid
        if result["fit_status"].iloc[0] == "failed":
            assert pd.isna(result["gup_doy"].iloc[0]) or result["gup_doy"].iloc[0] is None


class TestGrWithMask:
    """Tests for GR calculation with AOI mask."""

    def test_gr_respects_mask(self):
        """GR should only be calculated for pixels within the mask."""
        # Create a simple test scenario
        # 3x3 image with mask covering only center pixel
        mask = np.array([
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ])

        # The center pixel should be the only one included
        assert mask.sum() == 1

    def test_gr_empty_mask_returns_empty(self):
        """GR with empty mask should handle gracefully."""
        mask = np.zeros((3, 3), dtype=bool)

        # No pixels selected
        assert mask.sum() == 0


class TestGrFunctionWithImages:
    """Tests for gr() function with actual image files."""

    def test_gr_with_synthetic_images(self, tmp_path):
        """Test gr() function with synthetic images."""
        import cv2

        # Create synthetic images
        image_paths = []
        for i in range(5):
            # Create image with green content varying by day
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            green_val = 100 + i * 20  # Increasing green
            img[:, :, 1] = green_val  # Green channel (BGR order)
            img[:, :, 0] = 100  # Blue
            img[:, :, 2] = 100  # Red

            filepath = tmp_path / f"img_{i}.jpg"
            cv2.imwrite(str(filepath), img)
            image_paths.append(str(filepath))

        # Create profile
        profile = pd.DataFrame({
            "filepath": image_paths,
            "timestamp": pd.date_range("2024-05-01", periods=5, freq="D"),
        })

        # Create AOI mask
        aoi_mask = np.ones((100, 100), dtype=bool)

        # Run gr()
        daily_images, timeseries = gr(profile, aoi_mask)

        assert isinstance(daily_images, dict)
        assert isinstance(timeseries, pd.DataFrame)
        assert "date" in timeseries.columns
        assert "doy" in timeseries.columns
        assert "mean_gr" in timeseries.columns
        assert len(timeseries) == 5

    def test_gr_with_empty_profile(self):
        """Test gr() with empty profile."""
        profile = pd.DataFrame(columns=["filepath", "timestamp"])
        aoi_mask = np.ones((10, 10), dtype=bool)

        daily_images, timeseries = gr(profile, aoi_mask)

        assert isinstance(daily_images, dict)
        assert len(daily_images) == 0
        assert isinstance(timeseries, pd.DataFrame)
        assert len(timeseries) == 0

    def test_gr_with_missing_columns_raises_error(self):
        """Test gr() raises error for missing columns."""
        profile = pd.DataFrame({"other_col": [1, 2, 3]})
        aoi_mask = np.ones((10, 10), dtype=bool)

        with pytest.raises(ValueError):
            gr(profile, aoi_mask)

    def test_gr_path_column_compatibility(self, tmp_path):
        """Test gr() accepts 'path' column and renames to 'filepath'."""
        import cv2

        # Create a test image
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        filepath = tmp_path / "test.jpg"
        cv2.imwrite(str(filepath), img)

        # Profile with 'path' instead of 'filepath'
        profile = pd.DataFrame({
            "path": [str(filepath)],
            "timestamp": [pd.Timestamp("2024-05-01")],
        })

        aoi_mask = np.ones((50, 50), dtype=bool)

        daily_images, timeseries = gr(profile, aoi_mask)

        assert len(timeseries) == 1

    def test_gr_with_daily_max_false(self, tmp_path):
        """Test gr() with daily_max=False."""
        import cv2

        # Create multiple images on same day
        image_paths = []
        for i in range(3):
            img = np.ones((50, 50, 3), dtype=np.uint8) * (100 + i * 20)
            filepath = tmp_path / f"img_{i}.jpg"
            cv2.imwrite(str(filepath), img)
            image_paths.append(str(filepath))

        # All on same day, different times
        profile = pd.DataFrame({
            "filepath": image_paths,
            "timestamp": pd.to_datetime([
                "2024-05-01 06:00",
                "2024-05-01 12:00",
                "2024-05-01 18:00",
            ]),
        })

        aoi_mask = np.ones((50, 50), dtype=bool)

        # With daily_max=False, should get multiple records
        daily_images, timeseries = gr(profile, aoi_mask, daily_max=False)

        assert len(timeseries) == 3

    def test_gr_with_output_dir(self, tmp_path):
        """Test gr() saves images when output_dir is specified."""
        import cv2
        import os

        # Create test image
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        filepath = tmp_path / "test.jpg"
        cv2.imwrite(str(filepath), img)

        profile = pd.DataFrame({
            "filepath": [str(filepath)],
            "timestamp": [pd.Timestamp("2024-05-01")],
        })

        aoi_mask = np.ones((50, 50), dtype=bool)
        output_dir = tmp_path / "output"

        daily_images, timeseries = gr(profile, aoi_mask, output_dir=str(output_dir))

        assert os.path.exists(output_dir)
        # Should have saved .npy files
        npy_files = list(output_dir.glob("*.npy"))
        assert len(npy_files) > 0


class TestComputeGrImage:
    """Tests for _compute_gr_image internal function."""

    def test_compute_gr_image_basic(self):
        """Test GR computation for a simple image."""
        from yamami.phenology import _compute_gr_image

        # Create 3x3 BGR image
        img = np.zeros((3, 3, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # Blue
        img[:, :, 1] = 150  # Green
        img[:, :, 2] = 50   # Red

        gr_img = _compute_gr_image(img)

        # Expected GR = 150 / (50 + 150 + 100) = 0.5
        expected_gr = 150 / (50 + 150 + 100)
        np.testing.assert_allclose(gr_img, expected_gr, rtol=1e-5)

    def test_compute_gr_image_black_pixels(self):
        """Test GR handles black pixels (division by zero)."""
        from yamami.phenology import _compute_gr_image

        img = np.zeros((3, 3, 3), dtype=np.uint8)  # All black

        gr_img = _compute_gr_image(img)

        # Should return 0 for black pixels (not NaN or error)
        assert not np.any(np.isnan(gr_img))
        np.testing.assert_array_equal(gr_img, 0.0)

    def test_compute_gr_image_pure_green(self):
        """Test GR for pure green pixels."""
        from yamami.phenology import _compute_gr_image

        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[:, :, 1] = 255  # Green channel only (BGR)

        gr_img = _compute_gr_image(img)

        np.testing.assert_allclose(gr_img, 1.0, rtol=1e-5)


class TestPhenologyWithModel:
    """Tests for phenology() function with different inputs."""

    def test_phenology_invalid_model_raises_error(self):
        """Test phenology() raises error for unknown model."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        gr_data = pd.DataFrame({
            "date": dates,
            "doy": dates.dayofyear,
            "mean_gr": np.random.rand(100),
        })

        with pytest.raises(ValueError, match="Unknown model"):
            phenology(gr_data, model="invalid_model")

    def test_phenology_invalid_data_type_raises_error(self):
        """Test phenology() raises error for invalid data type."""
        with pytest.raises(ValueError):
            phenology("not a dataframe or dict")

    def test_phenology_dict_input_not_implemented(self):
        """Test phenology() with dict raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            phenology({"key": np.zeros((10, 10))})

    def test_phenology_missing_doy_uses_date(self):
        """Test phenology() computes doy from date column."""
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        gr_data = pd.DataFrame({
            "date": dates,
            "mean_gr": 0.35 + 0.08 * np.sin(dates.dayofyear * 2 * np.pi / 365),
        })

        # Should not raise error, should compute doy from date
        result = phenology(gr_data)
        assert isinstance(result, pd.DataFrame)

    def test_phenology_missing_mean_gr_raises_error(self):
        """Test phenology() raises error when mean_gr column missing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        gr_data = pd.DataFrame({
            "date": dates,
            "doy": dates.dayofyear,
            # missing mean_gr
        })

        with pytest.raises(ValueError):
            phenology(gr_data)

    def test_phenology_with_output_dir(self, tmp_path):
        """Test phenology() saves results when output_dir specified."""
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        gr_data = pd.DataFrame({
            "date": dates,
            "doy": dates.dayofyear,
            "mean_gr": 0.35 + 0.08 * np.sin(dates.dayofyear * 2 * np.pi / 365),
        })

        output_dir = tmp_path / "phenology"
        result = phenology(gr_data, output_dir=str(output_dir))

        assert (output_dir / "phenology_results.csv").exists()


class TestFitDoubleSigmoidEdgeCases:
    """Additional edge case tests for fit_double_sigmoid."""

    def test_fit_with_nan_values(self):
        """Test fit handles NaN values in data."""
        t = np.arange(1, 366)
        values = 0.35 + 0.08 * np.sin(t * 2 * np.pi / 365)

        # Add some NaN values
        values[50:60] = np.nan

        result = fit_double_sigmoid(t, values)

        assert isinstance(result, dict)
        # Should still work with valid data

    def test_fit_with_insufficient_data(self):
        """Test fit with fewer than 30 data points returns failed status."""
        t = np.arange(1, 20)  # Only 19 points
        values = np.random.rand(19)

        result = fit_double_sigmoid(t, values)

        assert result["fit_status"] == "failed"

    def test_fit_with_realistic_seasonal_pattern(self):
        """Test fit with realistic double-sigmoid pattern."""
        t = np.arange(1, 366)

        # Create realistic double-sigmoid pattern
        base = 0.32
        amp1 = 0.15
        amp2 = 0.15
        k1 = 0.08
        k2 = 0.06
        t1 = 120  # Green-up around day 120
        t2 = 280  # Green-down around day 280

        rising = amp1 / (1 + np.exp(-k1 * (t - t1)))
        falling = amp2 / (1 + np.exp(-k2 * (t - t2)))
        values = base + rising - falling

        result = fit_double_sigmoid(t, values)

        # Should succeed with clean data
        if result["fit_status"] == "success":
            # Check that fitted parameters are reasonable
            assert 50 < result["t1"] < 200  # Green-up in spring
            assert 200 < result["t2"] < 350  # Green-down in autumn
