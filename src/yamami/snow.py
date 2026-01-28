"""
Snow detection and snowmelt date estimation module.

This module provides functionality for:
- Extracting snow masks from images using threshold-based classification
- Detecting high-reflectance rocks to exclude from snow analysis
- Estimating pixel-level snowmelt day-of-year

The snow detection uses RGB sum thresholding with optional Otsu method
for automatic threshold determination. Snowmelt DOY is estimated using
multi-stage temporal analysis to find when snow disappears for
consecutive days.
"""

from typing import Optional

import numpy as np
import pandas as pd

from yamami.logging import get_logger

logger = get_logger(__name__)


def snow(
    profile: pd.DataFrame,
    aoi_mask: np.ndarray,
    otsu: bool = True,
    output_dir: Optional[str] = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Extract snow masks from images using threshold-based classification.

    Uses RGB sum thresholding to classify pixels as snow or non-snow.
    When otsu=True, the threshold is determined automatically using
    Otsu's method on the grayscale (RGB sum) values.

    Args:
        profile: DataFrame with columns 'filepath', 'doy', and RGB mean values
                ('R_mean', 'G_mean', 'B_mean').
        aoi_mask: Binary mask defining the area of interest.
                  Shape should match expected image dimensions.
        otsu: If True, use Otsu's method for automatic thresholding.
              If False, use a fixed threshold. Default is True.
        output_dir: Optional directory to save intermediate results.

    Returns:
        tuple: (masks, thresholds) where:
            - masks: dict mapping DOY (int) to binary snow mask (np.ndarray)
            - thresholds: DataFrame with 'filepath' and 'threshold' columns
    """
    if len(profile) == 0:
        logger.info("Empty profile provided, returning empty results")
        return {}, pd.DataFrame(columns=["filepath", "threshold"])

    masks = {}
    threshold_records = []

    for _, row in profile.iterrows():
        filepath = row["filepath"]
        doy = int(row["doy"])

        # Calculate RGB sum as brightness indicator
        rgb_sum = row["R_mean"] + row["G_mean"] + row["B_mean"]

        # Determine threshold
        if otsu:
            threshold = _calculate_otsu_threshold(profile)
        else:
            # Fixed threshold: typical snow has RGB sum > 600
            threshold = 600.0

        # Create snow mask based on threshold
        # For profile-based analysis, we create a uniform mask
        # based on the average brightness of the image
        snow_mask = np.zeros(aoi_mask.shape, dtype=np.uint8)
        if rgb_sum >= threshold:
            # Image appears to have significant snow coverage
            snow_mask = aoi_mask.copy()
        else:
            # Image appears to have little or no snow
            snow_mask = np.zeros(aoi_mask.shape, dtype=np.uint8)

        # Apply AOI mask
        snow_mask = snow_mask * aoi_mask

        masks[doy] = snow_mask
        threshold_records.append({
            "filepath": filepath,
            "threshold": threshold,
        })

        logger.debug(
            f"Processed DOY {doy}: RGB sum={rgb_sum:.1f}, "
            f"threshold={threshold:.1f}, snow_pixels={snow_mask.sum()}"
        )

    thresholds = pd.DataFrame(threshold_records)

    if output_dir:
        _save_snow_results(masks, thresholds, output_dir)

    logger.info(f"Snow detection completed for {len(masks)} images")
    return masks, thresholds


def _calculate_otsu_threshold(profile: pd.DataFrame) -> float:
    """
    Calculate Otsu's threshold from profile RGB values.

    Uses the RGB sum distribution across all images to find
    an optimal threshold that separates snow from non-snow images.

    Args:
        profile: DataFrame with R_mean, G_mean, B_mean columns.

    Returns:
        float: Optimal threshold value.
    """
    if len(profile) == 0:
        return 600.0  # Default threshold

    # Calculate RGB sums for all images
    rgb_sums = (
        profile["R_mean"] + profile["G_mean"] + profile["B_mean"]
    ).values

    if len(rgb_sums) < 2:
        # Not enough data for Otsu, return default
        return 600.0

    # Otsu's method implementation
    # Create histogram
    min_val = rgb_sums.min()
    max_val = rgb_sums.max()

    if max_val - min_val < 1:
        # No variance, return midpoint
        return (min_val + max_val) / 2

    # Normalize to 0-255 range for histogram
    normalized = ((rgb_sums - min_val) / (max_val - min_val) * 255).astype(int)
    normalized = np.clip(normalized, 0, 255)

    # Compute histogram
    hist, _ = np.histogram(normalized, bins=256, range=(0, 256))
    hist = hist.astype(float)

    # Normalize histogram
    total = hist.sum()
    if total == 0:
        return 600.0

    hist = hist / total

    # Compute cumulative sums
    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))

    # Global mean
    global_mean = cum_mean[-1]

    # Between-class variance
    variance = np.zeros(256)
    for t in range(256):
        w0 = cum_sum[t]
        w1 = 1 - w0

        if w0 == 0 or w1 == 0:
            continue

        mu0 = cum_mean[t] / w0 if w0 > 0 else 0
        mu1 = (global_mean - cum_mean[t]) / w1 if w1 > 0 else 0

        variance[t] = w0 * w1 * (mu0 - mu1) ** 2

    # Find optimal threshold
    optimal_t = np.argmax(variance)

    # Convert back to original scale
    threshold = min_val + (optimal_t / 255) * (max_val - min_val)

    logger.debug(f"Otsu threshold calculated: {threshold:.1f}")
    return threshold


def detect_high_reflectance_rock(
    image: np.ndarray,
    aoi_mask: np.ndarray,
) -> np.ndarray:
    """
    Detect high-reflectance rocks to exclude from snow analysis.

    High-reflectance rocks can be confused with snow because they
    also appear bright in RGB images. This function identifies
    rock pixels based on their high but stable reflectance pattern.

    Args:
        image: RGB image as numpy array with shape (H, W, 3).
        aoi_mask: Binary mask defining the area of interest.

    Returns:
        np.ndarray: Binary mask of rock pixels (1 = rock, 0 = not rock).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB with shape (H, W, 3)")

    if image.shape[:2] != aoi_mask.shape:
        raise ValueError("Image and AOI mask must have matching dimensions")

    # Calculate brightness as sum of RGB channels
    brightness = image.sum(axis=2).astype(float)

    # High reflectance threshold (normalized)
    max_possible = 255 * 3
    high_threshold = 0.7 * max_possible  # 70% of maximum brightness

    # Detect high-reflectance pixels
    high_reflectance = brightness >= high_threshold

    # Rock detection heuristics:
    # 1. High reflectance
    # 2. Low color saturation (rocks are often grayish)

    # Calculate saturation-like metric (difference between max and min channel)
    max_channel = image.max(axis=2).astype(float)
    min_channel = image.min(axis=2).astype(float)
    color_range = max_channel - min_channel

    # Low saturation indicates rock (gray-ish)
    low_saturation = color_range < 30  # Small color range

    # Combine criteria
    rock_mask = (high_reflectance & low_saturation).astype(np.uint8)

    # Apply AOI mask
    rock_mask = rock_mask * aoi_mask

    logger.debug(f"Detected {rock_mask.sum()} rock pixels")
    return rock_mask


def snowmelt(
    snow_masks: dict,
    aoi_mask: np.ndarray,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Estimate pixel-level snowmelt day-of-year.

    Analyzes temporal series of snow masks to determine when snow
    melts at each pixel. Snowmelt is defined as the first date where
    snow disappears for consecutive days (to avoid false positives
    from temporary cloud cover).

    Args:
        snow_masks: dict mapping DOY (int) to binary snow mask (np.ndarray).
                   Keys should be sorted chronologically.
        aoi_mask: Binary mask defining the area of interest.
        output_dir: Optional directory to save results.

    Returns:
        pd.DataFrame: Snowmelt results with columns:
            - row: Pixel row index
            - col: Pixel column index
            - doy: Day of year when snow melted
            - confidence: Confidence score (0-1)
            - status: 'melted', 'no_snow', 'persistent_snow', or 'uncertain'
    """
    if len(snow_masks) == 0:
        logger.info("Empty snow masks provided, returning empty results")
        return pd.DataFrame(columns=["row", "col", "doy", "confidence", "status"])

    # Get sorted DOY list
    doys = sorted(snow_masks.keys())

    # Get mask dimensions from first mask
    first_mask = snow_masks[doys[0]]
    height, width = first_mask.shape

    # Find AOI pixels
    aoi_rows, aoi_cols = np.where(aoi_mask > 0)

    results = []

    for row, col in zip(aoi_rows, aoi_cols):
        # Extract time series for this pixel
        snow_series = [snow_masks[doy][row, col] for doy in doys]

        # Analyze time series
        status, melt_doy, confidence = _analyze_pixel_snowmelt(
            doys, snow_series
        )

        results.append({
            "row": row,
            "col": col,
            "doy": melt_doy,
            "confidence": confidence,
            "status": status,
        })

    df = pd.DataFrame(results)

    if output_dir:
        _save_snowmelt_results(df, output_dir)

    # Log summary statistics
    status_counts = df["status"].value_counts()
    logger.info(f"Snowmelt analysis completed: {len(df)} pixels analyzed")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count} pixels")

    return df


def _analyze_pixel_snowmelt(
    doys: list,
    snow_series: list,
    min_consecutive: int = 2,
) -> tuple[str, float, float]:
    """
    Analyze a single pixel's snow time series to determine snowmelt date.

    Args:
        doys: List of DOY values (sorted).
        snow_series: List of snow presence values (1=snow, 0=no snow).
        min_consecutive: Minimum consecutive snow-free days required.

    Returns:
        tuple: (status, melt_doy, confidence)
    """
    snow_series = np.array(snow_series)

    # Check if pixel ever had snow
    if snow_series.sum() == 0:
        return "no_snow", np.nan, 1.0

    # Check if snow persists throughout
    if snow_series[-min_consecutive:].sum() == min_consecutive:
        return "persistent_snow", np.nan, 1.0

    # Find snowmelt date (first time snow is absent for consecutive days)
    melt_doy = None
    for i in range(len(doys)):
        if snow_series[i] == 0:
            # Check if snow stays gone for min_consecutive days
            remaining = snow_series[i:i + min_consecutive]
            if len(remaining) >= min_consecutive and remaining.sum() == 0:
                # Verify snow doesn't return significantly later
                future_snow = snow_series[i + min_consecutive:]
                if future_snow.sum() == 0:
                    melt_doy = doys[i]
                    break

    if melt_doy is None:
        # Snow pattern is uncertain (intermittent)
        # Use last transition from snow to no-snow
        for i in range(len(doys) - 1, -1, -1):
            if snow_series[i] == 0 and (i == 0 or snow_series[i - 1] == 1):
                melt_doy = doys[i]
                return "uncertain", melt_doy, 0.5

        return "uncertain", np.nan, 0.0

    # Calculate confidence based on data quality
    # Higher confidence with more data points and clearer transition
    n_observations = len(doys)
    confidence = min(1.0, n_observations / 10)  # Scale with data amount

    return "melted", melt_doy, confidence


def _save_snow_results(
    masks: dict,
    thresholds: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save snow detection results to output directory."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Save thresholds
    thresholds.to_csv(
        os.path.join(output_dir, "snow_thresholds.csv"),
        index=False,
    )

    # Save masks as numpy arrays
    masks_dir = os.path.join(output_dir, "snow_masks")
    os.makedirs(masks_dir, exist_ok=True)

    for doy, mask in masks.items():
        np.save(os.path.join(masks_dir, f"snow_mask_doy{doy:03d}.npy"), mask)

    logger.info(f"Snow results saved to {output_dir}")


def _save_snowmelt_results(df: pd.DataFrame, output_dir: str) -> None:
    """Save snowmelt results to output directory."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(
        os.path.join(output_dir, "snowmelt.csv"),
        index=False,
    )

    logger.info(f"Snowmelt results saved to {output_dir}")
