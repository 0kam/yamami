"""
Unit tests for align module.

Tests phase correlation change detection and homography estimation
using synthetic data.
"""

import numpy as np
import pandas as pd
import pytest


class TestPhaseCorrelation:
    """Tests for phase correlation change detection."""

    def test_phase_correlation_no_shift(self):
        """Test phase correlation detects no shift for identical images."""
        from yamami.align import phase_correlation_change_detect

        # Two identical images
        img1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = img1.copy()

        shift = phase_correlation_change_detect(img1, img2)

        assert abs(shift[0]) < 1.0
        assert abs(shift[1]) < 1.0

    def test_phase_correlation_detects_horizontal_shift(self):
        """Test phase correlation detects horizontal shift."""
        from yamami.align import phase_correlation_change_detect

        # Create image with distinct features
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img1[40:60, 40:60] = 255  # White square

        # Shift horizontally by 10 pixels
        img2 = np.zeros((100, 100), dtype=np.uint8)
        img2[40:60, 50:70] = 255

        shift = phase_correlation_change_detect(img1, img2)

        # Should detect roughly 10 pixel horizontal shift
        assert abs(shift[0] - 10) < 2.0

    def test_phase_correlation_detects_vertical_shift(self):
        """Test phase correlation detects vertical shift."""
        from yamami.align import phase_correlation_change_detect

        # Create image with distinct features
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img1[40:60, 40:60] = 255

        # Shift vertically by 10 pixels
        img2 = np.zeros((100, 100), dtype=np.uint8)
        img2[50:70, 40:60] = 255

        shift = phase_correlation_change_detect(img1, img2)

        # Should detect roughly 10 pixel vertical shift
        assert abs(shift[1] - 10) < 2.0


class TestSegmentTimeseries:
    """Tests for time series segmentation."""

    def test_segment_timeseries_single_segment(self):
        """Test segmentation returns single segment for stable series."""
        from yamami.align import segment_timeseries

        # No significant shifts
        shifts = np.zeros((20, 2))

        segments = segment_timeseries(shifts, threshold=5.0)

        assert len(segments) == 1
        assert segments[0]["start_idx"] == 0
        assert segments[0]["end_idx"] == 19

    def test_segment_timeseries_detects_change(self):
        """Test segmentation detects change point."""
        from yamami.align import segment_timeseries

        # Stable, then shift, then stable
        shifts = np.zeros((20, 2))
        shifts[10:, 0] = 20  # Shift after index 10

        segments = segment_timeseries(shifts, threshold=5.0)

        assert len(segments) >= 2

    def test_segment_timeseries_multiple_segments(self):
        """Test segmentation handles multiple changes."""
        from yamami.align import segment_timeseries

        shifts = np.zeros((30, 2))
        shifts[10:20, 0] = 20  # Shift in middle
        shifts[20:, 0] = 0  # Back to original

        segments = segment_timeseries(shifts, threshold=5.0)

        assert len(segments) >= 2


class TestSelectAnchorImages:
    """Tests for anchor image selection."""

    def test_select_anchor_single_segment(self):
        """Test anchor selection for single segment."""
        from yamami.align import select_anchor_images

        segments = [{"segment_id": 0, "start_idx": 0, "end_idx": 9}]
        profile = pd.DataFrame(
            {
                "filepath": [f"img_{i}.jpg" for i in range(10)],
                "brightness": [0.5] * 10,
            }
        )

        anchors = select_anchor_images(segments, profile)

        assert len(anchors) == 1
        assert anchors[0]["segment_id"] == 0

    def test_select_anchor_prefers_bright_images(self):
        """Test anchor selection prefers images with good brightness."""
        from yamami.align import select_anchor_images

        segments = [{"segment_id": 0, "start_idx": 0, "end_idx": 4}]
        profile = pd.DataFrame(
            {
                "filepath": [f"img_{i}.jpg" for i in range(5)],
                "brightness": [0.3, 0.4, 0.6, 0.5, 0.2],  # img_2 is best
            }
        )

        anchors = select_anchor_images(segments, profile)

        assert anchors[0]["filepath"] == "img_2.jpg"


class TestHomography:
    """Tests for homography estimation."""

    def test_estimate_homography_identity(self):
        """Test homography estimation for matched points."""
        from yamami.align import estimate_homography

        # Identical point pairs (should give identity transform)
        src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst_pts = src_pts.copy()

        H = estimate_homography(src_pts, dst_pts)

        # Should be close to identity matrix
        identity = np.eye(3)
        assert np.allclose(H, identity, atol=0.1)

    def test_estimate_homography_translation(self):
        """Test homography estimation for translation."""
        from yamami.align import estimate_homography

        src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        # Translate by (10, 20)
        dst_pts = src_pts + np.array([10, 20])

        H = estimate_homography(src_pts, dst_pts)

        # Check translation components
        assert abs(H[0, 2] - 10) < 1.0  # tx
        assert abs(H[1, 2] - 20) < 1.0  # ty

    def test_estimate_homography_insufficient_points(self):
        """Test homography returns None for insufficient points."""
        from yamami.align import estimate_homography

        # Only 3 points (need at least 4)
        src_pts = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32)
        dst_pts = src_pts.copy()

        H = estimate_homography(src_pts, dst_pts)

        assert H is None


class TestApplyWarp:
    """Tests for image warping."""

    def test_apply_warp_identity(self):
        """Test warp with identity transform returns same image."""
        from yamami.align import apply_warp

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float32)

        warped = apply_warp(img, H)

        # Should be essentially the same
        assert warped.shape == img.shape
        # Allow small differences due to interpolation
        assert np.mean(np.abs(warped.astype(float) - img.astype(float))) < 5.0

    def test_apply_warp_changes_image(self):
        """Test warp with translation changes image."""
        from yamami.align import apply_warp

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = 255  # White square in center

        # Translation matrix (shift by 10, 10)
        H = np.array([[1, 0, 10], [0, 1, 10], [0, 0, 1]], dtype=np.float32)

        warped = apply_warp(img, H)

        # Original center should now be darker
        assert np.mean(warped[40:60, 40:60]) < np.mean(img[40:60, 40:60])


class TestAlignFunction:
    """Tests for main align() function."""

    def test_align_creates_output_dir(self, sample_profile_df, tmp_path):
        """Test align creates output directory."""
        from yamami.align import align
        import os

        ref_image = sample_profile_df["filepath"].iloc[0]
        output_dir = str(tmp_path / "aligned")

        _, _, aligned_dir = align(sample_profile_df, ref=ref_image, output_dir=output_dir)

        assert os.path.exists(aligned_dir)

    def test_align_accepts_method_parameter(self, sample_profile_df, tmp_path):
        """Test align accepts method parameter."""
        from yamami.align import align

        ref_image = sample_profile_df["filepath"].iloc[0]

        # Should not raise error with method parameter
        result = align(
            sample_profile_df,
            ref=ref_image,
            method="roma",
            output_dir=str(tmp_path / "aligned"),
        )
        assert result is not None

    def test_align_accepts_max_rounds_parameter(self, sample_profile_df, tmp_path):
        """Test align accepts max_rounds parameter."""
        from yamami.align import align

        ref_image = sample_profile_df["filepath"].iloc[0]

        # Should not raise error with max_rounds parameter
        result = align(
            sample_profile_df,
            ref=ref_image,
            max_rounds=5,
            output_dir=str(tmp_path / "aligned"),
        )
        assert result is not None


# Fixtures
@pytest.fixture
def sample_profile_df(tmp_path):
    """Create a sample profile DataFrame for testing."""
    import cv2

    image_paths = []
    for i in range(5):
        img_path = tmp_path / f"test_image_{i}.jpg"
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        image_paths.append(str(img_path))

    return pd.DataFrame(
        {
            "filepath": image_paths,
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "brightness": [0.5] * 5,
        }
    )
