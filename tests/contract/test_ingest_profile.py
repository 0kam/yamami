"""
Contract tests for ingest() and profile() functions.

These tests verify that the functions return the expected types and structures,
ensuring API compatibility.
"""

import pandas as pd
import pytest

from yamami.ingest import ingest, profile


class TestIngestContract:
    """Contract tests for ingest() function (T011)."""

    def test_ingest_returns_dataframe(self, tmp_path):
        """ingest() should return a pandas DataFrame."""
        # Create a test directory with an image file
        test_image = tmp_path / "photo_20240501_0800.jpg"
        test_image.write_bytes(b"\xff\xd8\xff")  # Minimal JPEG header

        result = ingest(str(tmp_path))

        assert isinstance(result, pd.DataFrame)

    def test_ingest_dataframe_has_path_column(self, tmp_path):
        """ingest() DataFrame should have 'path' column."""
        # Create a test directory with an image file
        test_image = tmp_path / "photo_20240501_0800.jpg"
        test_image.write_bytes(b"\xff\xd8\xff")  # Minimal JPEG header

        result = ingest(str(tmp_path))

        assert "path" in result.columns

    def test_ingest_works_with_directory_containing_images(self, tmp_path):
        """ingest() should work with a directory containing images."""
        # Create multiple test images
        for i in range(3):
            test_image = tmp_path / f"photo_2024050{i}_0800.jpg"
            test_image.write_bytes(b"\xff\xd8\xff")

        result = ingest(str(tmp_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_ingest_returns_empty_dataframe_for_empty_directory(self, tmp_path):
        """ingest() should return empty DataFrame for directory without images."""
        result = ingest(str(tmp_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestProfileContract:
    """Contract tests for profile() function (T012)."""

    def test_profile_returns_dataframe(self, tmp_path, sample_image):
        """profile() should return a pandas DataFrame."""
        import cv2

        # Create a test image
        test_image_path = tmp_path / "photo_20240501_0800.jpg"
        cv2.imwrite(str(test_image_path), sample_image)

        index = ingest(str(tmp_path))
        result = profile(index)

        assert isinstance(result, pd.DataFrame)

    def test_profile_has_required_columns(self, tmp_path, sample_image):
        """profile() DataFrame should have all required columns."""
        import cv2

        # Create a test image
        test_image_path = tmp_path / "photo_20240501_0800.jpg"
        cv2.imwrite(str(test_image_path), sample_image)

        index = ingest(str(tmp_path))
        result = profile(index)

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
            assert col in result.columns, f"Missing column: {col}"

    def test_profile_handles_empty_index(self):
        """profile() should handle empty DataFrame gracefully."""
        empty_index = pd.DataFrame(columns=["path"])
        result = profile(empty_index)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_profile_extracts_timestamp(self, tmp_path, sample_image):
        """profile() should extract timestamp from filename."""
        import cv2
        from datetime import datetime

        # Create a test image with datetime in filename
        test_image_path = tmp_path / "photo_20240501_0800.jpg"
        cv2.imwrite(str(test_image_path), sample_image)

        index = ingest(str(tmp_path))
        result = profile(index)

        assert result.iloc[0]["timestamp"] == datetime(2024, 5, 1, 8, 0)

    def test_profile_handles_unparseable_filename(self, tmp_path, sample_image):
        """profile() should handle filenames without datetime pattern."""
        import cv2

        # Create a test image without datetime in filename
        test_image_path = tmp_path / "random_image.jpg"
        cv2.imwrite(str(test_image_path), sample_image)

        index = ingest(str(tmp_path))
        result = profile(index)

        # timestamp should be None for unparseable filename
        assert result.iloc[0]["timestamp"] is None

    def test_profile_timestamp_source_filename(self, tmp_path, sample_image):
        """profile() with timestamp_source='filename' should use filename."""
        import cv2
        from datetime import datetime

        test_image_path = tmp_path / "photo_20240501_0800.jpg"
        cv2.imwrite(str(test_image_path), sample_image)

        index = ingest(str(tmp_path))
        result = profile(index, timestamp_source="filename")

        assert result.iloc[0]["timestamp"] == datetime(2024, 5, 1, 8, 0)

    def test_profile_custom_filename_pattern(self, tmp_path, sample_image):
        """profile() should support custom filename pattern."""
        import cv2
        from datetime import datetime

        # Create image with non-standard datetime format
        test_image_path = tmp_path / "IMG_DATE20240501.jpg"
        cv2.imwrite(str(test_image_path), sample_image)

        index = ingest(str(tmp_path))
        result = profile(index, filename_pattern=r"DATE(?P<date>\d{8})")

        assert result.iloc[0]["timestamp"] == datetime(2024, 5, 1, 0, 0)
