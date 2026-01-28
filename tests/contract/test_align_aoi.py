"""
Contract tests for align and aoi functions.

These tests verify the API contracts:
- align() returns tuple (DataFrame segments, DataFrame warp_params, str aligned_dir)
- aoi() returns tuple (DataFrame skyline, ndarray aoi_mask)
"""

import numpy as np
import pandas as pd
import pytest


class TestAlignContract:
    """Contract tests for align() function."""

    def test_align_returns_tuple(self, sample_profile_df, tmp_path):
        """align() must return a tuple of length 3."""
        from yamami.align import align

        ref_image = sample_profile_df["filepath"].iloc[0]
        result = align(
            sample_profile_df, ref=ref_image, output_dir=str(tmp_path / "aligned")
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_align_first_element_is_dataframe(self, sample_profile_df, tmp_path):
        """First element (segments) must be a DataFrame."""
        from yamami.align import align

        ref_image = sample_profile_df["filepath"].iloc[0]
        segments, _, _ = align(
            sample_profile_df, ref=ref_image, output_dir=str(tmp_path / "aligned")
        )

        assert isinstance(segments, pd.DataFrame)

    def test_align_second_element_is_dataframe(self, sample_profile_df, tmp_path):
        """Second element (warp_params) must be a DataFrame."""
        from yamami.align import align

        ref_image = sample_profile_df["filepath"].iloc[0]
        _, warp_params, _ = align(
            sample_profile_df, ref=ref_image, output_dir=str(tmp_path / "aligned")
        )

        assert isinstance(warp_params, pd.DataFrame)

    def test_align_third_element_is_string(self, sample_profile_df, tmp_path):
        """Third element (aligned_dir) must be a string."""
        from yamami.align import align

        ref_image = sample_profile_df["filepath"].iloc[0]
        _, _, aligned_dir = align(
            sample_profile_df, ref=ref_image, output_dir=str(tmp_path / "aligned")
        )

        assert isinstance(aligned_dir, str)

    def test_align_segments_has_required_columns(self, sample_profile_df, tmp_path):
        """Segments DataFrame must have required columns."""
        from yamami.align import align

        ref_image = sample_profile_df["filepath"].iloc[0]
        segments, _, _ = align(
            sample_profile_df, ref=ref_image, output_dir=str(tmp_path / "aligned")
        )

        # Segments should have segment_id and boundaries
        assert "segment_id" in segments.columns
        assert "start_idx" in segments.columns
        assert "end_idx" in segments.columns

    def test_align_warp_params_has_required_columns(self, sample_profile_df, tmp_path):
        """Warp params DataFrame must have required columns."""
        from yamami.align import align

        ref_image = sample_profile_df["filepath"].iloc[0]
        _, warp_params, _ = align(
            sample_profile_df, ref=ref_image, output_dir=str(tmp_path / "aligned")
        )

        # Warp params should have filepath and transformation info
        assert "filepath" in warp_params.columns


class TestAOIContract:
    """Contract tests for aoi() function."""

    def test_aoi_returns_tuple(self, sample_ref_image, tmp_path):
        """aoi() must return a tuple of length 2."""
        from yamami.aoi import aoi

        result = aoi(sample_ref_image, output_dir=str(tmp_path / "aoi"))

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_aoi_first_element_is_dataframe(self, sample_ref_image, tmp_path):
        """First element (skyline) must be a DataFrame."""
        from yamami.aoi import aoi

        skyline, _ = aoi(sample_ref_image, output_dir=str(tmp_path / "aoi"))

        assert isinstance(skyline, pd.DataFrame)

    def test_aoi_second_element_is_ndarray(self, sample_ref_image, tmp_path):
        """Second element (aoi_mask) must be a numpy array."""
        from yamami.aoi import aoi

        _, aoi_mask = aoi(sample_ref_image, output_dir=str(tmp_path / "aoi"))

        assert isinstance(aoi_mask, np.ndarray)

    def test_aoi_skyline_has_required_columns(self, sample_ref_image, tmp_path):
        """Skyline DataFrame must have x and y columns."""
        from yamami.aoi import aoi

        skyline, _ = aoi(sample_ref_image, output_dir=str(tmp_path / "aoi"))

        assert "x" in skyline.columns
        assert "y" in skyline.columns

    def test_aoi_mask_is_binary(self, sample_ref_image, tmp_path):
        """AOI mask must be a binary mask (0 or 255 values only)."""
        from yamami.aoi import aoi

        _, aoi_mask = aoi(sample_ref_image, output_dir=str(tmp_path / "aoi"))

        unique_values = np.unique(aoi_mask)
        assert all(v in [0, 255] for v in unique_values)


# Fixtures for contract tests
@pytest.fixture
def sample_profile_df(tmp_path):
    """Create a sample profile DataFrame for testing."""
    import cv2

    image_paths = []
    for i in range(5):
        img_path = tmp_path / f"test_image_{i}.jpg"
        # Create a minimal image file
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        image_paths.append(str(img_path))

    return pd.DataFrame(
        {
            "filepath": image_paths,
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "site": ["MRD"] * 5,
            "azimuth": ["N"] * 5,
            "camera": ["NikL"] * 5,
            "band": ["RGB"] * 5,
        }
    )


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
