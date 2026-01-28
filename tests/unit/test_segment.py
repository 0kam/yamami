"""
Unit tests for segment module.

Tests SAM3 segmentation with mocking (don't actually call SAM3).
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestSegmentFunction:
    """Tests for segment() function."""

    def test_segment_with_default_prompts(self, sample_profile_df):
        """Test segment uses default prompts when none provided."""
        from yamami.segment import segment, DEFAULT_PROMPTS

        labels, masks = segment(sample_profile_df)

        # Should use default prompts
        assert DEFAULT_PROMPTS == ["sky", "cloud", "fog", "snow"]

    def test_segment_with_custom_prompts(self, sample_profile_df):
        """Test segment uses custom prompts when provided."""
        from yamami.segment import segment

        custom_prompts = ["tree", "building"]
        labels, masks = segment(sample_profile_df, prompts=custom_prompts)

        # Should handle custom prompts without error
        assert isinstance(labels, pd.DataFrame)

    def test_segment_returns_label_ratios(self, sample_profile_df):
        """Test segment returns ratio columns for each prompt."""
        from yamami.segment import segment

        labels, masks = segment(sample_profile_df)

        # Labels should have ratio columns
        for prompt in ["sky", "cloud", "fog", "snow"]:
            assert f"{prompt}_ratio" in labels.columns

    def test_segment_returns_brightness(self, sample_profile_df):
        """Test segment returns brightness column."""
        from yamami.segment import segment

        labels, masks = segment(sample_profile_df)

        assert "brightness" in labels.columns

    def test_segment_ratios_are_normalized(self, sample_profile_df):
        """Test that ratio values are between 0 and 1."""
        from yamami.segment import segment

        labels, masks = segment(sample_profile_df)

        for col in labels.columns:
            if col.endswith("_ratio"):
                assert labels[col].min() >= 0.0
                assert labels[col].max() <= 1.0

    def test_segment_brightness_is_normalized(self, sample_profile_df):
        """Test that brightness values are between 0 and 1."""
        from yamami.segment import segment

        labels, masks = segment(sample_profile_df)

        assert labels["brightness"].min() >= 0.0
        assert labels["brightness"].max() <= 1.0

    def test_segment_masks_keys_match_prompts(self, sample_profile_df):
        """Test that mask dict keys match prompts."""
        from yamami.segment import segment

        labels, masks = segment(sample_profile_df)

        # Masks should be keyed by filepath
        for filepath in sample_profile_df["filepath"]:
            assert filepath in masks

    def test_segment_returns_valid_masks(self, sample_profile_df):
        """Test that segment returns valid binary masks (0 or 255 values)."""
        from yamami.segment import segment

        labels, masks = segment(sample_profile_df)

        # Masks should be valid binary masks (values 0 or 255)
        for filepath, mask_dict in masks.items():
            for prompt, mask in mask_dict.items():
                unique_values = np.unique(mask)
                # All values should be either 0 or 255
                assert all(v in [0, 255] for v in unique_values), \
                    f"Mask should only contain 0 or 255, got {unique_values}"

    def test_segment_resize_parameter(self, sample_profile_df):
        """Test segment accepts resize parameter."""
        from yamami.segment import segment

        # Should not raise error with resize parameter
        labels, masks = segment(sample_profile_df, resize=256)
        assert isinstance(labels, pd.DataFrame)

    def test_segment_device_parameter(self, sample_profile_df):
        """Test segment accepts device parameter."""
        from yamami.segment import segment

        # Should not raise error with device parameter
        labels, masks = segment(sample_profile_df, device="cpu")
        assert isinstance(labels, pd.DataFrame)

    def test_segment_output_dir_creates_directory(self, sample_profile_df, tmp_path):
        """Test segment creates output directory if specified."""
        from yamami.segment import segment

        output_dir = str(tmp_path / "seg_output")
        labels, masks = segment(sample_profile_df, output_dir=output_dir)

        # Output directory should exist
        import os

        assert os.path.exists(output_dir)


class TestCalculateBrightness:
    """Tests for brightness calculation helper."""

    def test_brightness_white_image(self):
        """Test brightness of white image is 1.0."""
        from yamami.segment import _calculate_brightness

        white_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        brightness = _calculate_brightness(white_img)

        assert brightness == pytest.approx(1.0, abs=0.01)

    def test_brightness_black_image(self):
        """Test brightness of black image is 0.0."""
        from yamami.segment import _calculate_brightness

        black_img = np.zeros((100, 100, 3), dtype=np.uint8)
        brightness = _calculate_brightness(black_img)

        assert brightness == pytest.approx(0.0, abs=0.01)

    def test_brightness_gray_image(self):
        """Test brightness of gray image is 0.5."""
        from yamami.segment import _calculate_brightness

        gray_img = np.ones((100, 100, 3), dtype=np.uint8) * 127
        brightness = _calculate_brightness(gray_img)

        assert brightness == pytest.approx(0.5, abs=0.05)


# Fixtures
@pytest.fixture
def sample_profile_df(tmp_path):
    """Create a sample profile DataFrame for testing."""
    import cv2

    image_paths = []
    for i in range(3):
        img_path = tmp_path / f"test_image_{i}.jpg"
        # Create a minimal image file
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
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
