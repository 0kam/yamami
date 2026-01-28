"""
Unit tests for yamami ingest module.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from yamami.ingest import (
    compute_luminance_stats,
    extract_exif,
    ingest,
)


class TestDirectoryScanner:
    """Tests for directory scanner functionality (T013)."""

    def test_scanner_finds_jpeg_files(self, tmp_path):
        """Scanner should find .jpg files."""
        # Create test files
        (tmp_path / "test1.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "test2.jpg").write_bytes(b"\xff\xd8\xff")

        result = ingest(str(tmp_path))

        assert len(result) == 2

    def test_scanner_finds_tiff_files(self, tmp_path):
        """Scanner should find .tif and .tiff files."""
        # Create test files
        (tmp_path / "test1.tif").write_bytes(b"II*\x00")  # Little-endian TIFF
        (tmp_path / "test2.tiff").write_bytes(b"MM\x00*")  # Big-endian TIFF

        result = ingest(str(tmp_path))

        assert len(result) == 2

    def test_scanner_ignores_non_image_extensions(self, tmp_path):
        """Scanner should ignore non-image files."""
        # Create test files
        (tmp_path / "test.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "test.txt").write_text("not an image")
        (tmp_path / "test.pdf").write_bytes(b"%PDF")
        (tmp_path / "test.py").write_text("# python file")

        result = ingest(str(tmp_path))

        assert len(result) == 1
        assert result.iloc[0]["path"].endswith(".jpg")

    def test_scanner_finds_uppercase_extensions(self, tmp_path):
        """Scanner should find files with uppercase extensions."""
        (tmp_path / "test.JPG").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "test.JPEG").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "test.TIF").write_bytes(b"II*\x00")

        result = ingest(str(tmp_path))

        assert len(result) == 3

    def test_scanner_finds_mixed_extensions(self, tmp_path):
        """Scanner should find files with both .jpg and .jpeg extensions."""
        (tmp_path / "test1.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "test2.jpeg").write_bytes(b"\xff\xd8\xff")

        result = ingest(str(tmp_path))

        assert len(result) == 2

    def test_scanner_custom_patterns(self, tmp_path):
        """Scanner should support custom file patterns."""
        (tmp_path / "test.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "test.png").write_bytes(b"\x89PNG")

        # Only look for PNG files
        result = ingest(str(tmp_path), patterns=["*.png"])

        assert len(result) == 1
        assert result.iloc[0]["path"].endswith(".png")

    def test_scanner_recursive_search(self, tmp_path):
        """Scanner should find files in subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.jpg").write_bytes(b"\xff\xd8\xff")
        (subdir / "nested.jpg").write_bytes(b"\xff\xd8\xff")

        result = ingest(str(tmp_path))

        assert len(result) == 2


class TestLuminanceStats:
    """Tests for luminance statistics computation (T014)."""

    def test_compute_luminance_stats_rgb_image(self):
        """Should compute correct stats for RGB image."""
        # Create a simple gray image (all pixels = 128)
        img = np.full((100, 100, 3), 128, dtype=np.uint8)

        stats = compute_luminance_stats(img)

        assert "mean_luma" in stats
        assert "std_luma" in stats
        assert "p05_luma" in stats
        assert "p50_luma" in stats
        assert "p95_luma" in stats

        # For uniform image, mean should be close to 128
        assert abs(stats["mean_luma"] - 128) < 1

        # For uniform image, std should be close to 0
        assert stats["std_luma"] < 1

    def test_compute_luminance_stats_grayscale_image(self):
        """Should compute correct stats for grayscale image."""
        # Create a grayscale gradient
        img = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            img[:, i] = int(i * 255 / 99)

        stats = compute_luminance_stats(img)

        # Mean should be around 127.5 (middle of gradient)
        assert 125 < stats["mean_luma"] < 130

        # Percentiles should be ordered
        assert stats["p05_luma"] < stats["p50_luma"] < stats["p95_luma"]

    def test_compute_luminance_stats_percentiles(self):
        """Should compute correct percentile values."""
        # Create image with known distribution
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 0  # First half: black
        img[50:, :] = 255  # Second half: white

        stats = compute_luminance_stats(img)

        # P50 should be around the boundary (either 0 or 255)
        # P05 should be 0 (black region)
        # P95 should be 255 (white region)
        assert stats["p05_luma"] == 0
        assert stats["p95_luma"] == 255

    def test_compute_luminance_stats_with_white_image(self):
        """Should handle all-white image."""
        img = np.full((100, 100, 3), 255, dtype=np.uint8)

        stats = compute_luminance_stats(img)

        assert abs(stats["mean_luma"] - 255) < 1
        assert stats["std_luma"] < 1

    def test_compute_luminance_stats_with_black_image(self):
        """Should handle all-black image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        stats = compute_luminance_stats(img)

        assert stats["mean_luma"] < 1
        assert stats["std_luma"] < 1


class TestEXIFExtraction:
    """Tests for EXIF extraction functionality (T015)."""

    def test_extract_exif_returns_dict(self, tmp_path, sample_image):
        """extract_exif should return a dictionary."""
        # Save image with OpenCV (no EXIF)
        test_path = tmp_path / "test.jpg"
        cv2.imwrite(str(test_path), sample_image)

        result = extract_exif(str(test_path))

        assert isinstance(result, dict)

    def test_extract_exif_handles_missing_exif(self, tmp_path, sample_image):
        """Should gracefully handle images without EXIF data."""
        # Save image with OpenCV (typically no EXIF)
        test_path = tmp_path / "test.jpg"
        cv2.imwrite(str(test_path), sample_image)

        result = extract_exif(str(test_path))

        # Should return empty dict, not raise exception
        assert isinstance(result, dict)

    def test_extract_exif_with_pillow_image(self, tmp_path):
        """Should extract EXIF from Pillow-saved images if present."""
        # Create image with Pillow
        img = Image.new("RGB", (100, 100), color="red")
        test_path = tmp_path / "test.jpg"
        img.save(str(test_path))

        result = extract_exif(str(test_path))

        assert isinstance(result, dict)

    def test_extract_exif_nonexistent_file(self, tmp_path):
        """Should return empty dict for non-existent file."""
        result = extract_exif(str(tmp_path / "nonexistent.jpg"))

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_extract_exif_non_image_file(self, tmp_path):
        """Should return empty dict for non-image file."""
        test_path = tmp_path / "test.txt"
        test_path.write_text("not an image")

        result = extract_exif(str(test_path))

        assert isinstance(result, dict)
        assert len(result) == 0
