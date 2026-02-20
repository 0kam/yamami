"""
Image ingestion and profiling module for yamami.

Provides functions for scanning directories for images and computing
metadata and luminance statistics.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS

from yamami.parser import parse_filename


# Default patterns for image file detection
DEFAULT_PATTERNS = ["*.jpg", "*.jpeg", "*.tif", "*.tiff"]

# Default luminance stats when image cannot be read
_EMPTY_LUMA_STATS = {
    "mean_luma": None,
    "std_luma": None,
    "p05_luma": None,
    "p50_luma": None,
    "p95_luma": None,
    "saturated_ratio": None,
}


def ingest(
    directory: str,
    patterns: Optional[list[str]] = None,
    timestamp_source: str = "filename",
    timestamp_pattern: Optional[str] = None,
) -> pd.DataFrame:
    """
    Scan directory for images and return DataFrame with metadata and statistics.

    Recursively scans the specified directory for image files matching
    the given patterns, then extracts metadata and luminance statistics
    for each image.

    Args:
        directory: Path to the directory to scan.
        patterns: List of glob patterns to match (e.g., ["*.jpg", "*.png"]).
                  Defaults to ["*.jpg", "*.jpeg", "*.tif", "*.tiff"].
        timestamp_source: Source for timestamp extraction.
                          - "filename": Extract from filename (default)
                          - "exif": Extract from EXIF metadata
        timestamp_pattern: Optional custom regex pattern for timestamp parsing.
                           Only used when timestamp_source="filename".

    Returns:
        pd.DataFrame: DataFrame with columns:
            - path: Absolute path to the image
            - filename: Base filename
            - timestamp: Datetime from filename or EXIF (or None)
            - width: Image width in pixels
            - height: Image height in pixels
            - filesize: File size in bytes
            - mean_luma: Mean luminance (0-255)
            - std_luma: Standard deviation of luminance
            - p05_luma: 5th percentile luminance
            - p50_luma: 50th percentile luminance
            - p95_luma: 95th percentile luminance
            - saturated_ratio: Ratio of saturated pixels (>=250)

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        NotADirectoryError: If the specified path is not a directory.

    Example:
        >>> profiles = ingest("/path/to/images")
        >>> print(profiles[["filename", "timestamp", "mean_luma"]])
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    directory_path = Path(directory)

    # Validate directory exists
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    # Collect image paths matching patterns
    image_paths = _collect_image_paths(directory_path, patterns)

    # Build profile for each image
    from tqdm import tqdm

    records = []
    for path in tqdm(image_paths, desc="ingest"):
        record = _build_image_record(
            path,
            timestamp_source=timestamp_source,
            timestamp_pattern=timestamp_pattern,
        )
        records.append(record)

    return pd.DataFrame(records)


def _collect_image_paths(directory_path: Path, patterns: list[str]) -> list[str]:
    """
    Collect image paths matching the given patterns.

    Args:
        directory_path: Path object for the directory to scan.
        patterns: List of glob patterns to match.

    Returns:
        Sorted list of unique absolute paths as strings.
    """
    image_paths = []

    for pattern in patterns:
        # Search recursively with case-insensitive matching
        image_paths.extend(directory_path.rglob(pattern))

        # Also search with uppercase pattern for case-insensitive matching
        upper_pattern = pattern.upper()
        if upper_pattern != pattern:
            image_paths.extend(directory_path.rglob(upper_pattern))

    # Remove duplicates and convert to absolute paths
    return sorted(set(str(p.resolve()) for p in image_paths))


def _build_image_record(
    path: str,
    timestamp_source: str,
    timestamp_pattern: Optional[str],
) -> dict:
    """
    Build a record dictionary for a single image.

    Args:
        path: Absolute path to the image file.
        timestamp_source: Source for timestamp ("filename" or "exif").
        timestamp_pattern: Optional regex pattern for filename parsing.

    Returns:
        Dictionary with image metadata and statistics.
    """
    filename = os.path.basename(path)

    # Extract timestamp
    if timestamp_source == "exif":
        exif = extract_exif(path)
        timestamp = _parse_exif_datetime(exif)
    else:
        timestamp = parse_filename(filename, pattern=timestamp_pattern)

    # Get file size
    try:
        filesize = os.path.getsize(path)
    except OSError:
        filesize = None

    # Read image and compute stats
    width, height, luma_stats = _read_image_stats(path)

    return {
        "path": path,
        "filename": filename,
        "timestamp": timestamp,
        "width": width,
        "height": height,
        "filesize": filesize,
        **luma_stats,
    }


def _read_image_stats(path: str) -> tuple[Optional[int], Optional[int], dict]:
    """
    Read image and compute dimensions and luminance statistics.

    Args:
        path: Path to the image file.

    Returns:
        Tuple of (width, height, luma_stats). Returns (None, None, empty_stats)
        if the image cannot be read.
    """
    try:
        img = cv2.imread(path)
        if img is not None:
            height, width = img.shape[:2]
            luma_stats = compute_luminance_stats(img)
            return width, height, luma_stats
    except (cv2.error, OSError):
        # cv2.error: OpenCV-specific errors
        # OSError: File system errors (permissions, etc.)
        pass

    return None, None, _EMPTY_LUMA_STATS.copy()


def compute_luminance_stats(image: np.ndarray) -> dict:
    """
    Compute luminance statistics for an image.

    Converts the image to grayscale (if color) and computes statistical
    measures of pixel intensity.

    Args:
        image: Input image as numpy array. Can be grayscale (H, W) or
               color (H, W, C) in BGR or RGB format.

    Returns:
        dict: Dictionary containing:
            - mean_luma: Mean luminance value (0-255)
            - std_luma: Standard deviation of luminance
            - p05_luma: 5th percentile luminance
            - p50_luma: 50th percentile (median) luminance
            - p95_luma: 95th percentile luminance
            - saturated_ratio: Ratio of saturated pixels (>=250) for backlight detection

    Example:
        >>> import cv2
        >>> img = cv2.imread("image.jpg")
        >>> stats = compute_luminance_stats(img)
        >>> print(f"Mean: {stats['mean_luma']:.2f}")
    """
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Flatten for statistical computations
    pixels = gray.flatten().astype(np.float64)

    # Calculate saturated pixel ratio (for backlight detection)
    saturated_count = np.sum(gray >= 250)
    saturated_ratio = float(saturated_count / gray.size)

    return {
        "mean_luma": float(np.mean(pixels)),
        "std_luma": float(np.std(pixels)),
        "p05_luma": float(np.percentile(pixels, 5)),
        "p50_luma": float(np.percentile(pixels, 50)),
        "p95_luma": float(np.percentile(pixels, 95)),
        "saturated_ratio": saturated_ratio,
    }


def extract_exif(path: str) -> dict:
    """
    Extract EXIF metadata from an image file.

    Uses Pillow to read EXIF data from the image. Returns an empty
    dictionary if the file doesn't exist, is not a valid image, or
    has no EXIF data.

    Args:
        path: Path to the image file.

    Returns:
        dict: Dictionary containing EXIF tag names and values.
              Returns empty dict if no EXIF data is available.

    Example:
        >>> exif = extract_exif("photo.jpg")
        >>> if "DateTimeOriginal" in exif:
        ...     print(f"Taken: {exif['DateTimeOriginal']}")
    """
    try:
        with Image.open(path) as img:
            exif_data = img._getexif()
            if exif_data is None:
                return {}

            # Convert numeric EXIF tags to human-readable names
            return {
                TAGS.get(tag, tag): value
                for tag, value in exif_data.items()
            }
    except Exception:
        # Return empty dict for any errors (file not found, not an image, etc.)
        return {}


def _parse_exif_datetime(exif: dict) -> Optional[datetime]:
    """
    Parse datetime from EXIF data.

    Looks for DateTimeOriginal, DateTimeDigitized, or DateTime tags.

    Args:
        exif: Dictionary of EXIF data from extract_exif().

    Returns:
        datetime or None: Parsed datetime if found and valid, None otherwise.
    """
    for tag in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
        if tag in exif:
            try:
                # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                return datetime.strptime(exif[tag], "%Y:%m:%d %H:%M:%S")
            except (ValueError, TypeError):
                continue
    return None
