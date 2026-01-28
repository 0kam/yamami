"""
Unit tests for qc module.

Tests QC thresholding logic.
"""

import numpy as np
import pandas as pd
import pytest


class TestPreQCFunction:
    """Tests for pre_qc() function."""

    def test_pre_qc_marks_dark_images(self):
        """Test pre_qc marks images with low brightness as unusable."""
        from yamami.qc import pre_qc

        profile = pd.DataFrame({"filepath": ["img1.jpg", "img2.jpg", "img3.jpg"]})
        labels = pd.DataFrame(
            {
                "filepath": ["img1.jpg", "img2.jpg", "img3.jpg"],
                "brightness": [0.05, 0.5, 0.7],  # First is too dark
                "cloud_ratio": [0.1, 0.1, 0.1],
                "fog_ratio": [0.1, 0.1, 0.1],
            }
        )

        result = pre_qc(profile, labels, brightness_min=0.08)

        assert result.loc[result["filepath"] == "img1.jpg", "is_usable"].iloc[0] == False
        assert "TOO_DARK" in result.loc[result["filepath"] == "img1.jpg", "reason_codes"].iloc[0]

    def test_pre_qc_marks_bright_images(self):
        """Test pre_qc marks images with too high brightness as unusable."""
        from yamami.qc import pre_qc

        profile = pd.DataFrame({"filepath": ["img1.jpg", "img2.jpg"]})
        labels = pd.DataFrame(
            {
                "filepath": ["img1.jpg", "img2.jpg"],
                "brightness": [0.5, 0.98],  # Second is too bright
                "cloud_ratio": [0.1, 0.1],
                "fog_ratio": [0.1, 0.1],
            }
        )

        result = pre_qc(profile, labels, brightness_max=0.95)

        assert result.loc[result["filepath"] == "img2.jpg", "is_usable"].iloc[0] == False
        assert "TOO_BRIGHT" in result.loc[result["filepath"] == "img2.jpg", "reason_codes"].iloc[0]

    def test_pre_qc_marks_foggy_images(self):
        """Test pre_qc marks images with high fog ratio as unusable."""
        from yamami.qc import pre_qc

        profile = pd.DataFrame({"filepath": ["img1.jpg", "img2.jpg"]})
        labels = pd.DataFrame(
            {
                "filepath": ["img1.jpg", "img2.jpg"],
                "brightness": [0.5, 0.5],
                "cloud_ratio": [0.1, 0.1],
                "fog_ratio": [0.1, 0.5],  # Second is too foggy
            }
        )

        result = pre_qc(profile, labels, fog_max=0.30)

        assert result.loc[result["filepath"] == "img2.jpg", "is_usable"].iloc[0] == False
        assert "HIGH_FOG" in result.loc[result["filepath"] == "img2.jpg", "reason_codes"].iloc[0]

    def test_pre_qc_marks_cloudy_images(self):
        """Test pre_qc marks images with high cloud ratio as unusable."""
        from yamami.qc import pre_qc

        profile = pd.DataFrame({"filepath": ["img1.jpg", "img2.jpg"]})
        labels = pd.DataFrame(
            {
                "filepath": ["img1.jpg", "img2.jpg"],
                "brightness": [0.5, 0.5],
                "cloud_ratio": [0.1, 0.6],  # Second is too cloudy
                "fog_ratio": [0.1, 0.1],
            }
        )

        result = pre_qc(profile, labels, cloud_max=0.40)

        assert result.loc[result["filepath"] == "img2.jpg", "is_usable"].iloc[0] == False
        assert "HIGH_CLOUD" in result.loc[result["filepath"] == "img2.jpg", "reason_codes"].iloc[0]

    def test_pre_qc_usable_image(self):
        """Test pre_qc marks good images as usable."""
        from yamami.qc import pre_qc

        profile = pd.DataFrame({"filepath": ["good_image.jpg"]})
        labels = pd.DataFrame(
            {
                "filepath": ["good_image.jpg"],
                "brightness": [0.5],
                "cloud_ratio": [0.1],
                "fog_ratio": [0.1],
            }
        )

        result = pre_qc(profile, labels)

        assert result.loc[0, "is_usable"] == True
        assert result.loc[0, "reason_codes"] == ""

    def test_pre_qc_multiple_reasons(self):
        """Test pre_qc can report multiple reasons."""
        from yamami.qc import pre_qc

        profile = pd.DataFrame({"filepath": ["bad_image.jpg"]})
        labels = pd.DataFrame(
            {
                "filepath": ["bad_image.jpg"],
                "brightness": [0.02],  # Too dark
                "cloud_ratio": [0.8],  # Too cloudy
                "fog_ratio": [0.5],  # Too foggy
            }
        )

        result = pre_qc(profile, labels)

        assert result.loc[0, "is_usable"] == False
        reason_codes = result.loc[0, "reason_codes"]
        assert "TOO_DARK" in reason_codes
        assert "HIGH_CLOUD" in reason_codes
        assert "HIGH_FOG" in reason_codes

    def test_pre_qc_default_thresholds(self):
        """Test pre_qc uses correct default thresholds."""
        from yamami.qc import pre_qc

        profile = pd.DataFrame({"filepath": ["img.jpg"]})
        labels = pd.DataFrame(
            {
                "filepath": ["img.jpg"],
                "brightness": [0.07],  # Just below default 0.08
                "cloud_ratio": [0.39],  # Just below default 0.40
                "fog_ratio": [0.29],  # Just below default 0.30
            }
        )

        result = pre_qc(profile, labels)

        # Should be marked as unusable because brightness is below threshold
        assert result.loc[0, "is_usable"] == False
        assert "TOO_DARK" in result.loc[0, "reason_codes"]


class TestQCFunction:
    """Tests for qc() function with AOI consideration."""

    def test_qc_marks_aoi_obscured(self):
        """Test qc marks images with obscured AOI as unusable."""
        from yamami.qc import qc

        profile = pd.DataFrame({"filepath": ["img1.jpg", "img2.jpg"]})
        labels = pd.DataFrame(
            {
                "filepath": ["img1.jpg", "img2.jpg"],
                "brightness": [0.5, 0.5],
                "cloud_ratio": [0.1, 0.1],
                "fog_ratio": [0.1, 0.1],
                "aoi_cloud_ratio": [0.1, 0.5],  # Second has obscured AOI
            }
        )
        aoi_mask = np.ones((100, 100), dtype=np.uint8) * 255

        result = qc(profile, labels, aoi_mask, aoi_cloud_max=0.20)

        assert result.loc[result["filepath"] == "img2.jpg", "is_usable"].iloc[0] == False
        assert "AOI_OBSCURED" in result.loc[result["filepath"] == "img2.jpg", "reason_codes"].iloc[0]

    def test_qc_usable_with_clear_aoi(self):
        """Test qc marks images with clear AOI as usable."""
        from yamami.qc import qc

        profile = pd.DataFrame({"filepath": ["good_image.jpg"]})
        labels = pd.DataFrame(
            {
                "filepath": ["good_image.jpg"],
                "brightness": [0.5],
                "cloud_ratio": [0.1],
                "fog_ratio": [0.1],
                "aoi_cloud_ratio": [0.1],
            }
        )
        aoi_mask = np.ones((100, 100), dtype=np.uint8) * 255

        result = qc(profile, labels, aoi_mask)

        assert result.loc[0, "is_usable"] == True

    def test_qc_inherits_pre_qc_failures(self):
        """Test qc includes pre_qc failure reasons."""
        from yamami.qc import qc

        profile = pd.DataFrame({"filepath": ["bad_image.jpg"]})
        labels = pd.DataFrame(
            {
                "filepath": ["bad_image.jpg"],
                "brightness": [0.02],  # Too dark
                "cloud_ratio": [0.1],
                "fog_ratio": [0.1],
                "aoi_cloud_ratio": [0.5],  # AOI obscured
            }
        )
        aoi_mask = np.ones((100, 100), dtype=np.uint8) * 255

        result = qc(profile, labels, aoi_mask)

        assert result.loc[0, "is_usable"] == False
        reason_codes = result.loc[0, "reason_codes"]
        assert "TOO_DARK" in reason_codes
        assert "AOI_OBSCURED" in reason_codes


class TestReasonCodes:
    """Tests for reason code constants."""

    def test_reason_codes_are_defined(self):
        """Test all reason codes are defined."""
        from yamami.qc import (
            REASON_TOO_DARK,
            REASON_TOO_BRIGHT,
            REASON_HIGH_FOG,
            REASON_HIGH_CLOUD,
            REASON_AOI_OBSCURED,
        )

        assert REASON_TOO_DARK == "TOO_DARK"
        assert REASON_TOO_BRIGHT == "TOO_BRIGHT"
        assert REASON_HIGH_FOG == "HIGH_FOG"
        assert REASON_HIGH_CLOUD == "HIGH_CLOUD"
        assert REASON_AOI_OBSCURED == "AOI_OBSCURED"
