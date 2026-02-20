"""
Area of Interest (AOI) extraction module for yamami.

Provides functions for detecting the skyline and creating AOI masks
for phenological analysis of ground vegetation.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from yamami.logging import get_logger

logger = get_logger(__name__)


def load_aoi_mask(aoi_mask_path: str) -> np.ndarray:
    """Load AOI mask from user-editable PNG.

    Reads the AOI mask PNG (saved by align()) and converts it to a boolean
    mask. Black pixels (0,0,0) are treated as excluded (False), all other
    pixels are included (True).

    Args:
        aoi_mask_path: Path to the aoi_mask.png file.

    Returns:
        Boolean mask array of shape (H, W) where True = AOI.

    Raises:
        FileNotFoundError: If the mask file does not exist.
        ValueError: If the mask file cannot be read or is invalid.
    """
    path = Path(aoi_mask_path)
    if not path.exists():
        raise FileNotFoundError(f"AOI mask not found: {aoi_mask_path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read AOI mask: {aoi_mask_path}")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"AOI mask must be a 3-channel image, got shape {img.shape}"
        )

    mask = np.any(img > 0, axis=2)
    logger.info(
        f"Loaded AOI mask from {path}: shape={mask.shape}, "
        f"AOI pixels={int(mask.sum())}"
    )
    return mask


def aoi(
    ref_image: str, output_dir: Optional[str] = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extract skyline and create AOI mask.

    .. deprecated::
        Use ``align()`` which now saves ``aoi_mask.png`` automatically,
        then load it with ``load_aoi_mask()``.

    Detects the skyline using blue channel gradient analysis and creates
    a binary mask where the AOI (area below skyline) is marked.

    Args:
        ref_image: Path to reference image for skyline detection.
        output_dir: Optional directory to save skyline and mask files.

    Returns:
        tuple: A tuple containing:
            - skyline (pd.DataFrame): DataFrame with 'x' and 'y' columns
              defining the skyline curve.
            - aoi_mask (np.ndarray): Binary mask (0/255) where 255 indicates
              the Area of Interest (below skyline).

    Example:
        >>> skyline, aoi_mask = aoi("ref_image.jpg", output_dir="output")
        >>> skyline.columns
        Index(['x', 'y'])
        >>> np.unique(aoi_mask)
        array([  0, 255], dtype=uint8)
    """
    # Create output directory if specified
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Read reference image
    img = cv2.imread(ref_image)
    if img is None:
        raise ValueError(f"Could not read reference image: {ref_image}")

    # Detect skyline
    skyline = _detect_skyline(img)

    # Generate AOI mask
    aoi_mask = _generate_aoi_mask(skyline, img.shape[:2])

    # Save outputs if directory specified
    if output_dir is not None:
        # Save skyline CSV
        skyline_path = Path(output_dir) / "skyline.csv"
        skyline.to_csv(skyline_path, index=False)
        logger.info(f"Saved skyline to {skyline_path}")

        # Save mask image
        mask_path = Path(output_dir) / "aoi_mask.png"
        cv2.imwrite(str(mask_path), aoi_mask)
        logger.info(f"Saved AOI mask to {mask_path}")

    logger.info(
        f"AOI extraction complete: skyline has {len(skyline)} points, "
        f"mask shape {aoi_mask.shape}"
    )

    return skyline, aoi_mask


def _detect_skyline(img: np.ndarray) -> pd.DataFrame:
    """
    Detect skyline using blue channel gradient analysis.

    The skyline is detected by finding the maximum vertical gradient
    in the blue channel for each column. Sky regions typically have
    higher blue values than ground regions.

    Args:
        img: Input image as numpy array (BGR format).

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns defining skyline.
    """
    # Extract blue channel (OpenCV uses BGR)
    blue = img[:, :, 0].astype(np.float32)

    # Calculate vertical gradient (Sobel)
    grad_y = cv2.Sobel(blue, cv2.CV_32F, 0, 1, ksize=5)

    # For each column, find the row with maximum negative gradient
    # (sky to ground transition)
    height, width = blue.shape
    skyline_y = np.zeros(width, dtype=np.int32)

    for x in range(width):
        column_grad = grad_y[:, x]

        # Find the row with maximum negative gradient (sky to ground)
        # Look in the upper 80% of the image to avoid bottom edges
        search_range = int(height * 0.8)

        # Find where gradient is most negative (transition from bright to dark)
        min_grad_idx = np.argmin(column_grad[:search_range])

        skyline_y[x] = min_grad_idx

    # Smooth the skyline to reduce noise
    skyline_y = _smooth_skyline(skyline_y, window_size=21)

    # Create DataFrame
    skyline = pd.DataFrame({"x": np.arange(width), "y": skyline_y})

    return skyline


def _smooth_skyline(skyline_y: np.ndarray, window_size: int = 21) -> np.ndarray:
    """
    Smooth skyline using median filter.

    Args:
        skyline_y: Array of y-coordinates for skyline.
        window_size: Size of median filter window.

    Returns:
        np.ndarray: Smoothed skyline y-coordinates.
    """
    # Apply median filter
    from scipy.ndimage import median_filter

    smoothed = median_filter(skyline_y.astype(np.float32), size=window_size)

    return smoothed.astype(np.int32)


def _generate_aoi_mask(skyline: pd.DataFrame, img_shape: tuple) -> np.ndarray:
    """
    Generate binary AOI mask from skyline.

    Creates a mask where pixels below the skyline (ground/vegetation)
    are marked as 255 (AOI) and pixels above (sky) are marked as 0.

    Args:
        skyline: DataFrame with 'x' and 'y' columns.
        img_shape: Tuple of (height, width) for output mask.

    Returns:
        np.ndarray: Binary mask (0/255) with dtype uint8.
    """
    height, width = img_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill below skyline
    for _, row in skyline.iterrows():
        x = int(row["x"])
        y = int(row["y"])

        if 0 <= x < width and 0 <= y < height:
            # Mark everything below skyline as AOI
            mask[y:, x] = 255

    return mask
