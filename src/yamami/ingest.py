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


def ingest(directory: str, patterns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Scan directory and return DataFrame with image paths.

    Recursively scans the specified directory for image files matching
    the given patterns (default: JPEG and TIFF files).

    Args:
        directory: Path to the directory to scan.
        patterns: List of glob patterns to match (e.g., ["*.jpg", "*.png"]).
                  Defaults to ["*.jpg", "*.jpeg", "*.tif", "*.tiff"].

    Returns:
        pd.DataFrame: DataFrame with a 'path' column containing absolute paths
                      to all matching image files.

    Example:
        >>> index = ingest("/path/to/images")
        >>> print(index.head())
                                                   path
        0  /path/to/images/mrd_085_eos_vis_20200420_1805_R.JPG
        1  /path/to/images/mrd_085_eos_vis_20200502_1605_R.JPG
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    directory_path = Path(directory)
    image_paths = []

    for pattern in patterns:
        # Search recursively with case-insensitive matching
        # First, search with the pattern as-is
        image_paths.extend(directory_path.rglob(pattern))

        # Also search with uppercase pattern for case-insensitive matching
        upper_pattern = pattern.upper()
        if upper_pattern != pattern:
            image_paths.extend(directory_path.rglob(upper_pattern))

    # Remove duplicates and convert to absolute paths
    unique_paths = sorted(set(str(p.resolve()) for p in image_paths))

    return pd.DataFrame({"path": unique_paths})


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

    return {
        "mean_luma": float(np.mean(pixels)),
        "std_luma": float(np.std(pixels)),
        "p05_luma": float(np.percentile(pixels, 5)),
        "p50_luma": float(np.percentile(pixels, 50)),
        "p95_luma": float(np.percentile(pixels, 95)),
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


def profile(
    index: pd.DataFrame,
    timestamp_source: str = "filename",
    filename_pattern: Optional[str] = None,
) -> pd.DataFrame:
    """
    Profile images and return DataFrame with metadata and statistics.

    For each image in the index, extracts:
    - Timestamp from filename or EXIF data
    - Image dimensions (width, height)
    - File size
    - Luminance statistics

    Args:
        index: DataFrame with 'path' column containing image paths.
               Typically the output of ingest().
        timestamp_source: Source for timestamp extraction.
                          - "filename": Extract from filename (default)
                          - "exif": Extract from EXIF metadata
        filename_pattern: Optional custom regex pattern for filename parsing.
                          Only used when timestamp_source="filename".
                          See parse_filename() for pattern format.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - path: Absolute path to the image
            - filename: Base filename
            - site: Site identifier (from filename, if available)
            - azimuth: Camera azimuth (from filename, if available)
            - camera: Camera identifier (from filename, if available)
            - band: Band identifier (from filename, if available)
            - timestamp: Datetime from filename or EXIF (or None)
            - width: Image width in pixels
            - height: Image height in pixels
            - filesize: File size in bytes
            - mean_luma: Mean luminance (0-255)
            - std_luma: Standard deviation of luminance
            - p05_luma: 5th percentile luminance
            - p50_luma: 50th percentile luminance
            - p95_luma: 95th percentile luminance

    Example:
        >>> index = ingest("/path/to/images")
        >>> profiles = profile(index)
        >>> print(profiles[["filename", "timestamp", "mean_luma"]])
    """
    columns = [
        "path",
        "filename",
        "site",
        "azimuth",
        "camera",
        "band",
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

    if index.empty:
        return pd.DataFrame(columns=columns)

    records = []

    for _, row in index.iterrows():
        path = row["path"]
        filename = os.path.basename(path)

        # Extract timestamp
        site = None
        azimuth = None
        camera = None
        band = None

        if timestamp_source == "exif":
            exif = extract_exif(path)
            timestamp = _parse_exif_datetime(exif)
        else:
            # Default: filename
            parsed = parse_filename(filename, pattern=filename_pattern)
            if parsed is not None:
                timestamp = parsed.get("timestamp")
                site = parsed.get("site")
                azimuth = parsed.get("azimuth")
                camera = parsed.get("camera")
                band = parsed.get("band")
            else:
                timestamp = None

        # Get file size
        try:
            filesize = os.path.getsize(path)
        except OSError:
            filesize = None

        # Read image and compute stats
        try:
            img = cv2.imread(path)
            if img is not None:
                height, width = img.shape[:2]
                luma_stats = compute_luminance_stats(img)
            else:
                width, height = None, None
                luma_stats = {
                    "mean_luma": None,
                    "std_luma": None,
                    "p05_luma": None,
                    "p50_luma": None,
                    "p95_luma": None,
                }
        except Exception:
            width, height = None, None
            luma_stats = {
                "mean_luma": None,
                "std_luma": None,
                "p05_luma": None,
                "p50_luma": None,
                "p95_luma": None,
            }

        record = {
            "path": path,
            "filename": filename,
            "site": site,
            "azimuth": azimuth,
            "camera": camera,
            "band": band,
            "timestamp": timestamp,
            "width": width,
            "height": height,
            "filesize": filesize,
            **luma_stats,
        }
        records.append(record)

    return pd.DataFrame(records)
