"""
Quality control module for yamami.

Provides pre-alignment and post-alignment quality control functions
to filter out unusable images based on brightness, fog, cloud coverage,
and AOI (Area of Interest) obscuration.
"""

import numpy as np
import pandas as pd

from yamami.logging import get_logger

logger = get_logger(__name__)

# Reason code constants
REASON_TOO_DARK = "TOO_DARK"
REASON_TOO_BRIGHT = "TOO_BRIGHT"
REASON_HIGH_FOG = "HIGH_FOG"
REASON_HIGH_CLOUD = "HIGH_CLOUD"
REASON_AOI_OBSCURED = "AOI_OBSCURED"


def pre_qc(
    profile: pd.DataFrame,
    labels: pd.DataFrame,
    brightness_min: float = 0.08,
    brightness_max: float = 0.95,
    fog_max: float = 0.30,
    cloud_max: float = 0.40,
) -> pd.DataFrame:
    """
    Pre-alignment quality control.

    Filters images based on brightness and atmospheric conditions before
    alignment. Images failing QC thresholds are marked as unusable with
    corresponding reason codes.

    Args:
        profile: DataFrame with 'filepath' column.
        labels: DataFrame with segmentation labels including 'filepath',
            'brightness', 'fog_ratio', and 'cloud_ratio' columns.
        brightness_min: Minimum acceptable brightness (0-1). Default is 0.08.
        brightness_max: Maximum acceptable brightness (0-1). Default is 0.95.
        fog_max: Maximum acceptable fog ratio (0-1). Default is 0.30.
        cloud_max: Maximum acceptable cloud ratio (0-1). Default is 0.40.

    Returns:
        pd.DataFrame: DataFrame with original profile data plus:
            - is_usable (bool): Whether the image passes QC.
            - reason_codes (str): Comma-separated list of failure reasons.

    Example:
        >>> result = pre_qc(profile, labels)
        >>> result[result["is_usable"] == False]["reason_codes"]
        0    TOO_DARK,HIGH_FOG
        Name: reason_codes, dtype: object
    """
    # Merge profile with labels on filepath
    merged = profile.merge(labels, on="filepath", how="left")

    # Initialize QC columns
    is_usable = pd.Series([True] * len(merged), index=merged.index)
    reason_codes = pd.Series([""] * len(merged), index=merged.index)

    # Check brightness thresholds
    if "brightness" in merged.columns:
        too_dark = merged["brightness"] < brightness_min
        too_bright = merged["brightness"] > brightness_max

        is_usable = is_usable & ~too_dark & ~too_bright
        reason_codes = _append_reason(reason_codes, too_dark, REASON_TOO_DARK)
        reason_codes = _append_reason(reason_codes, too_bright, REASON_TOO_BRIGHT)

    # Check fog ratio
    if "fog_ratio" in merged.columns:
        high_fog = merged["fog_ratio"] > fog_max
        is_usable = is_usable & ~high_fog
        reason_codes = _append_reason(reason_codes, high_fog, REASON_HIGH_FOG)

    # Check cloud ratio
    if "cloud_ratio" in merged.columns:
        high_cloud = merged["cloud_ratio"] > cloud_max
        is_usable = is_usable & ~high_cloud
        reason_codes = _append_reason(reason_codes, high_cloud, REASON_HIGH_CLOUD)

    # Build result DataFrame
    result = profile.copy()
    result["is_usable"] = is_usable.values
    result["reason_codes"] = reason_codes.values

    usable_count = result["is_usable"].sum()
    logger.info(
        f"Pre-QC: {usable_count}/{len(result)} images passed "
        f"(brightness=[{brightness_min:.2f},{brightness_max:.2f}], "
        f"fog<{fog_max:.2f}, cloud<{cloud_max:.2f})"
    )

    return result


def qc(
    profile: pd.DataFrame,
    labels: pd.DataFrame,
    aoi_mask: np.ndarray,
    brightness_min: float = 0.08,
    brightness_max: float = 0.95,
    fog_max: float = 0.30,
    cloud_max: float = 0.40,
    aoi_cloud_max: float = 0.20,
) -> pd.DataFrame:
    """
    Full quality control with AOI consideration.

    Performs pre-alignment QC checks plus additional checks for AOI
    (Area of Interest) obscuration. Use this after alignment when
    the AOI mask is available.

    Args:
        profile: DataFrame with 'filepath' column.
        labels: DataFrame with segmentation labels including 'filepath',
            'brightness', 'fog_ratio', 'cloud_ratio', and optionally
            'aoi_cloud_ratio' columns.
        aoi_mask: Binary mask (0/255) defining the Area of Interest.
        brightness_min: Minimum acceptable brightness (0-1). Default is 0.08.
        brightness_max: Maximum acceptable brightness (0-1). Default is 0.95.
        fog_max: Maximum acceptable fog ratio (0-1). Default is 0.30.
        cloud_max: Maximum acceptable cloud ratio (0-1). Default is 0.40.
        aoi_cloud_max: Maximum acceptable cloud ratio within AOI (0-1).
            Default is 0.20.

    Returns:
        pd.DataFrame: DataFrame with original profile data plus:
            - is_usable (bool): Whether the image passes QC.
            - reason_codes (str): Comma-separated list of failure reasons.

    Example:
        >>> result = qc(profile, labels, aoi_mask)
        >>> result[result["is_usable"] == False]["reason_codes"]
        0    AOI_OBSCURED
        Name: reason_codes, dtype: object
    """
    # First apply pre_qc checks
    result = pre_qc(
        profile,
        labels,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        fog_max=fog_max,
        cloud_max=cloud_max,
    )

    # Check AOI cloud coverage if column exists
    if "aoi_cloud_ratio" in labels.columns:
        # Merge with labels for AOI-specific checks
        merged = result.merge(labels[["filepath", "aoi_cloud_ratio"]], on="filepath", how="left")
        aoi_obscured = merged["aoi_cloud_ratio"] > aoi_cloud_max

        # Update is_usable
        result["is_usable"] = result["is_usable"] & ~aoi_obscured.values

        # Append reason codes
        for idx in merged[aoi_obscured].index:
            current = result.loc[idx, "reason_codes"]
            if current:
                result.loc[idx, "reason_codes"] = f"{current},{REASON_AOI_OBSCURED}"
            else:
                result.loc[idx, "reason_codes"] = REASON_AOI_OBSCURED

    usable_count = result["is_usable"].sum()
    logger.info(
        f"Full QC: {usable_count}/{len(result)} images passed "
        f"(aoi_cloud<{aoi_cloud_max:.2f})"
    )

    return result


def _append_reason(
    reason_codes: pd.Series, condition: pd.Series, reason: str
) -> pd.Series:
    """
    Append a reason code to entries where condition is True.

    Args:
        reason_codes: Series of current reason code strings.
        condition: Boolean series indicating which entries to update.
        reason: Reason code to append.

    Returns:
        pd.Series: Updated reason codes series.
    """
    result = reason_codes.copy()
    for idx in result[condition].index:
        current = result[idx]
        if current:
            result[idx] = f"{current},{reason}"
        else:
            result[idx] = reason
    return result
