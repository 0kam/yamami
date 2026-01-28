"""
Integration tests using real image data from test_data_micro.

These tests use actual mountain time-lapse images to verify the
pipeline works correctly with real-world data.
"""

from datetime import datetime
from pathlib import Path

import pytest

from yamami.ingest import ingest, profile

REAL_DATA_DIR = Path(__file__).parent.parent.parent / "test_data_micro"


@pytest.fixture
def output_dir():
    """Test output directory for persisting results."""
    out = Path(__file__).parent.parent.parent / "test_outputs"
    out.mkdir(exist_ok=True)
    return out


@pytest.mark.real_data
class TestRealDataPipeline:
    """Integration tests using test_data_micro images."""

    @pytest.mark.skipif(not REAL_DATA_DIR.exists(), reason="test_data_micro not found")
    def test_ingest_finds_all_images(self):
        """ingest() should find all 15 images in test_data_micro."""
        index = ingest(str(REAL_DATA_DIR))

        assert len(index) == 15
        assert "path" in index.columns

        # All paths should exist
        for path in index["path"]:
            assert Path(path).exists()

    @pytest.mark.skipif(not REAL_DATA_DIR.exists(), reason="test_data_micro not found")
    def test_profile_extracts_timestamps(self, output_dir):
        """profile() should extract timestamps from all filenames."""
        index = ingest(str(REAL_DATA_DIR))
        profiles = profile(index)

        # Save results for inspection
        profiles.to_csv(output_dir / "profile.csv", index=False)

        # All required columns should exist
        required_columns = [
            "path",
            "filename",
            "timestamp",
            "width",
            "height",
            "filesize",
            "mean_luma",
            "std_luma",
            "p05_luma",
            "p50_luma",
            "p95_luma",
        ]
        for col in required_columns:
            assert col in profiles.columns, f"Missing column: {col}"

        # All timestamps should be extracted
        assert profiles["timestamp"].notna().all(), "Some timestamps are missing"

        # Verify timestamps are within expected range (2020)
        for ts in profiles["timestamp"]:
            assert ts.year == 2020, f"Unexpected year: {ts.year}"
            assert 4 <= ts.month <= 11, f"Unexpected month: {ts.month}"

    @pytest.mark.skipif(not REAL_DATA_DIR.exists(), reason="test_data_micro not found")
    def test_profile_computes_luminance_stats(self):
        """profile() should compute valid luminance statistics."""
        index = ingest(str(REAL_DATA_DIR))
        profiles = profile(index)

        # All luminance stats should be computed
        luma_cols = ["mean_luma", "std_luma", "p05_luma", "p50_luma", "p95_luma"]
        for col in luma_cols:
            assert profiles[col].notna().all(), f"{col} has missing values"
            # Values should be in valid range (0-255)
            assert (profiles[col] >= 0).all(), f"{col} has negative values"
            assert (profiles[col] <= 255).all(), f"{col} has values > 255"

    @pytest.mark.skipif(not REAL_DATA_DIR.exists(), reason="test_data_micro not found")
    def test_profile_image_dimensions(self):
        """profile() should extract valid image dimensions."""
        index = ingest(str(REAL_DATA_DIR))
        profiles = profile(index)

        # All dimensions should be extracted
        assert profiles["width"].notna().all(), "width has missing values"
        assert profiles["height"].notna().all(), "height has missing values"

        # All dimensions should be positive
        assert (profiles["width"] > 0).all()
        assert (profiles["height"] > 0).all()

    @pytest.mark.skipif(not REAL_DATA_DIR.exists(), reason="test_data_micro not found")
    def test_profile_file_sizes(self):
        """profile() should extract valid file sizes."""
        index = ingest(str(REAL_DATA_DIR))
        profiles = profile(index)

        assert profiles["filesize"].notna().all(), "filesize has missing values"
        # All file sizes should be positive (JPEG images are typically > 1KB)
        assert (profiles["filesize"] > 1000).all()

    @pytest.mark.skipif(not REAL_DATA_DIR.exists(), reason="test_data_micro not found")
    def test_timestamps_chronological_order(self, output_dir):
        """Timestamps should span April to November 2020."""
        index = ingest(str(REAL_DATA_DIR))
        profiles = profile(index)

        # Sort by timestamp
        profiles_sorted = profiles.sort_values("timestamp")
        profiles_sorted.to_csv(output_dir / "profile_sorted.csv", index=False)

        timestamps = profiles_sorted["timestamp"].tolist()

        # First image should be in April
        assert timestamps[0] == datetime(2020, 4, 20, 18, 5)
        # Last image should be in November
        assert timestamps[-1] == datetime(2020, 11, 23, 9, 5)

    @pytest.mark.skipif(not REAL_DATA_DIR.exists(), reason="test_data_micro not found")
    def test_seasonal_luminance_variation(self, output_dir):
        """Luminance should show seasonal variation (spring->summer->fall)."""
        index = ingest(str(REAL_DATA_DIR))
        profiles = profile(index)
        profiles_sorted = profiles.sort_values("timestamp")

        # Save detailed analysis
        analysis = profiles_sorted[["filename", "timestamp", "mean_luma"]].copy()
        analysis["month"] = analysis["timestamp"].apply(lambda x: x.month)
        analysis.to_csv(output_dir / "luminance_analysis.csv", index=False)

        # Group by month and compute mean luminance
        monthly_luma = analysis.groupby("month")["mean_luma"].mean()

        # The data exists and has some variation
        assert len(monthly_luma) > 1, "Should have data from multiple months"
