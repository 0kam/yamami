"""
Contract tests for segment and QC functions.

These tests verify the API contracts:
- segment() returns tuple (DataFrame labels, dict of mask arrays)
- pre_qc() returns DataFrame with is_usable column
- qc() returns DataFrame with is_usable and reason_codes columns
"""

import numpy as np
import pandas as pd
import pytest


class TestSegmentContract:
    """Contract tests for segment() function."""

    def test_segment_returns_tuple(self, sample_profile_df):
        """segment() must return a tuple of (DataFrame, dict)."""
        from yamami.segment import segment

        result = segment(sample_profile_df)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_segment_first_element_is_dataframe(self, sample_profile_df):
        """First element of segment() result must be a DataFrame."""
        from yamami.segment import segment

        labels, _ = segment(sample_profile_df)

        assert isinstance(labels, pd.DataFrame)

    def test_segment_second_element_is_dict(self, sample_profile_df):
        """Second element of segment() result must be a dict of mask arrays."""
        from yamami.segment import segment

        _, masks = segment(sample_profile_df)

        assert isinstance(masks, dict)

    def test_segment_labels_has_required_columns(self, sample_profile_df):
        """Labels DataFrame must have required columns."""
        from yamami.segment import segment

        labels, _ = segment(sample_profile_df)

        # Labels should have filepath and segmentation label columns
        assert "filepath" in labels.columns

    def test_segment_masks_values_are_arrays(self, sample_profile_df):
        """Mask dict values must be dicts of numpy arrays keyed by filepath."""
        from yamami.segment import segment

        _, masks = segment(sample_profile_df)

        # masks is keyed by filepath, each value is a dict of prompt->mask
        for filepath, prompt_masks in masks.items():
            assert isinstance(prompt_masks, dict), f"Mask for {filepath} is not dict"
            for prompt, mask in prompt_masks.items():
                assert isinstance(mask, np.ndarray), f"Mask for {filepath}/{prompt} is not ndarray"


class TestPreQCContract:
    """Contract tests for pre_qc() function."""

    def test_pre_qc_returns_dataframe(self, sample_profile_df, sample_labels_df):
        """pre_qc() must return a DataFrame."""
        from yamami.qc import pre_qc

        result = pre_qc(sample_profile_df, sample_labels_df)

        assert isinstance(result, pd.DataFrame)

    def test_pre_qc_has_is_usable_column(self, sample_profile_df, sample_labels_df):
        """pre_qc() result must have is_usable column."""
        from yamami.qc import pre_qc

        result = pre_qc(sample_profile_df, sample_labels_df)

        assert "is_usable" in result.columns

    def test_pre_qc_is_usable_is_boolean(self, sample_profile_df, sample_labels_df):
        """is_usable column must be boolean type."""
        from yamami.qc import pre_qc

        result = pre_qc(sample_profile_df, sample_labels_df)

        assert result["is_usable"].dtype == bool

    def test_pre_qc_has_reason_codes_column(self, sample_profile_df, sample_labels_df):
        """pre_qc() result must have reason_codes column."""
        from yamami.qc import pre_qc

        result = pre_qc(sample_profile_df, sample_labels_df)

        assert "reason_codes" in result.columns


class TestQCContract:
    """Contract tests for qc() function."""

    def test_qc_returns_dataframe(
        self, sample_profile_df, sample_labels_df, sample_aoi_mask
    ):
        """qc() must return a DataFrame."""
        from yamami.qc import qc

        result = qc(sample_profile_df, sample_labels_df, sample_aoi_mask)

        assert isinstance(result, pd.DataFrame)

    def test_qc_has_is_usable_column(
        self, sample_profile_df, sample_labels_df, sample_aoi_mask
    ):
        """qc() result must have is_usable column."""
        from yamami.qc import qc

        result = qc(sample_profile_df, sample_labels_df, sample_aoi_mask)

        assert "is_usable" in result.columns

    def test_qc_has_reason_codes_column(
        self, sample_profile_df, sample_labels_df, sample_aoi_mask
    ):
        """qc() result must have reason_codes column."""
        from yamami.qc import qc

        result = qc(sample_profile_df, sample_labels_df, sample_aoi_mask)

        assert "reason_codes" in result.columns


# Fixtures for contract tests
@pytest.fixture
def sample_profile_df(tmp_path):
    """Create a sample profile DataFrame for testing."""
    # Create dummy image files
    image_paths = []
    for i in range(3):
        img_path = tmp_path / f"test_image_{i}.jpg"
        # Create a minimal image file
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        import cv2

        cv2.imwrite(str(img_path), img)
        image_paths.append(str(img_path))

    return pd.DataFrame(
        {
            "filepath": image_paths,
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "site": ["MRD"] * 3,
            "azimuth": ["N"] * 3,
            "camera": ["NikL"] * 3,
            "band": ["RGB"] * 3,
        }
    )


@pytest.fixture
def sample_labels_df(sample_profile_df):
    """Create a sample labels DataFrame for testing."""
    return pd.DataFrame(
        {
            "filepath": sample_profile_df["filepath"].tolist(),
            "sky_ratio": [0.3, 0.35, 0.25],
            "cloud_ratio": [0.1, 0.15, 0.2],
            "fog_ratio": [0.05, 0.1, 0.08],
            "snow_ratio": [0.0, 0.0, 0.0],
            "brightness": [0.5, 0.6, 0.4],
            "aoi_cloud_ratio": [0.1, 0.1, 0.1],  # For full QC testing
        }
    )


@pytest.fixture
def sample_aoi_mask():
    """Create a sample AOI mask for testing."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:100, :] = 255  # Lower half is AOI
    return mask
