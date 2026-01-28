"""
Vegetation phenology analysis module for YAMAMI.

This module provides functions for computing greenness ratio (GR) time series
and extracting phenological dates using double-sigmoid curve fitting.

The main functions are:
- gr(): Compute greenness ratio from RGB images
- fit_double_sigmoid(): Fit double-sigmoid curve to time series
- phenology(): Extract phenological dates (green-up, green-max, green-down)

Example usage:
    >>> from yamami.phenology import gr, phenology
    >>> images, timeseries = gr(profile_df, aoi_mask)
    >>> pheno_dates = phenology(timeseries)
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from yamami.logging import get_logger

logger = get_logger(__name__)


def _double_sigmoid(
    t: np.ndarray,
    base: float,
    amp1: float,
    k1: float,
    t1: float,
    amp2: float,
    k2: float,
    t2: float,
) -> np.ndarray:
    """
    Double-sigmoid model for vegetation phenology.

    The model describes seasonal vegetation dynamics with:
    - Rising sigmoid for spring green-up
    - Falling sigmoid for autumn senescence

    Model: f(t) = base + amp1/(1+exp(-k1*(t-t1))) - amp2/(1+exp(-k2*(t-t2)))

    Args:
        t: Time values (typically day of year).
        base: Baseline/winter value.
        amp1: Amplitude of rising sigmoid.
        k1: Rate parameter of rising sigmoid.
        t1: Inflection point (DOY) of rising sigmoid (green-up date).
        amp2: Amplitude of falling sigmoid.
        k2: Rate parameter of falling sigmoid.
        t2: Inflection point (DOY) of falling sigmoid (green-down date).

    Returns:
        Model predictions at time points t.
    """
    rising = amp1 / (1 + np.exp(-k1 * (t - t1)))
    falling = amp2 / (1 + np.exp(-k2 * (t - t2)))
    return base + rising - falling


def _compute_gr_image(image: np.ndarray) -> np.ndarray:
    """
    Compute greenness ratio for an RGB image.

    GR = G / (R + G + B)

    Args:
        image: RGB image with shape (H, W, 3), values in range [0, 255].

    Returns:
        Greenness ratio image with shape (H, W), values in range [0, 1].
    """
    # Convert to float to avoid integer overflow
    img_float = image.astype(np.float64)

    # Extract channels (assuming BGR order from OpenCV)
    b = img_float[:, :, 0]
    g = img_float[:, :, 1]
    r = img_float[:, :, 2]

    # Compute sum with handling for zero
    total = r + g + b

    # Compute GR, handling division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        gr = np.where(total > 0, g / total, 0.0)

    return gr


def gr(
    profile: pd.DataFrame,
    aoi_mask: np.ndarray,
    daily_max: bool = True,
    output_dir: Optional[str] = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Compute greenness ratio time series from image profile.

    Calculates GR = G / (R + G + B) for each pixel within the AOI mask,
    optionally aggregating to daily maximum values.

    Args:
        profile: DataFrame with 'filepath' and 'timestamp' columns.
        aoi_mask: Boolean mask array defining the area of interest.
            Pixels with True are included in the analysis.
        daily_max: If True, aggregate multiple observations per day to
            daily maximum value. Default is True.
        output_dir: Optional directory to save intermediate GR images.

    Returns:
        Tuple of (daily_images, timeseries) where:
        - daily_images: Dict mapping date strings to GR image arrays
        - timeseries: DataFrame with date, doy, and mean_gr columns

    Raises:
        ValueError: If profile is empty or missing required columns.
    """
    if profile.empty:
        logger.warning("Empty profile DataFrame provided")
        return {}, pd.DataFrame(columns=["date", "doy", "mean_gr"])

    required_cols = ["filepath", "timestamp"]
    missing_cols = [c for c in required_cols if c not in profile.columns]
    if missing_cols:
        # Try alternative column names
        if "path" in profile.columns and "filepath" not in profile.columns:
            profile = profile.rename(columns={"path": "filepath"})
        else:
            raise ValueError(f"Missing required columns: {missing_cols}")

    # Initialize storage
    gr_records = []
    daily_images = {}

    # Process each image
    for _, row in profile.iterrows():
        filepath = row["filepath"]
        timestamp = row["timestamp"]

        # Load image
        img = cv2.imread(str(filepath))
        if img is None:
            logger.warning(f"Could not load image: {filepath}")
            continue

        # Compute GR for the image
        gr_img = _compute_gr_image(img)

        # Apply AOI mask and compute mean
        if aoi_mask.shape != gr_img.shape:
            logger.warning(
                f"Mask shape {aoi_mask.shape} does not match image shape {gr_img.shape}"
            )
            continue

        masked_gr = gr_img[aoi_mask]
        if len(masked_gr) == 0:
            logger.warning("No pixels in AOI mask")
            continue

        mean_gr = np.nanmean(masked_gr)
        date = timestamp.date() if hasattr(timestamp, "date") else pd.Timestamp(timestamp).date()

        gr_records.append(
            {
                "timestamp": timestamp,
                "date": date,
                "mean_gr": mean_gr,
                "gr_image": gr_img,
            }
        )

    if not gr_records:
        logger.warning("No valid GR values computed")
        return {}, pd.DataFrame(columns=["date", "doy", "mean_gr"])

    # Create DataFrame
    df = pd.DataFrame(gr_records)

    if daily_max:
        # Group by date and take maximum
        daily_df = df.groupby("date").agg({"mean_gr": "max"}).reset_index()

        # Also get the corresponding image for each day's max
        for date in daily_df["date"].unique():
            day_data = df[df["date"] == date]
            max_idx = day_data["mean_gr"].idxmax()
            daily_images[str(date)] = day_data.loc[max_idx, "gr_image"]
    else:
        daily_df = df[["date", "mean_gr"]].copy()
        for i, row in df.iterrows():
            key = f"{row['date']}_{row['timestamp']}"
            daily_images[key] = row["gr_image"]

    # Add day of year
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["doy"] = daily_df["date"].dt.dayofyear

    # Save intermediate images if requested
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for date_str, gr_img in daily_images.items():
            # Save as float32 npy file
            np.save(output_path / f"gr_{date_str}.npy", gr_img.astype(np.float32))

    # Prepare output timeseries
    timeseries = daily_df[["date", "doy", "mean_gr"]].copy()
    timeseries = timeseries.sort_values("doy").reset_index(drop=True)

    return daily_images, timeseries


def fit_double_sigmoid(
    time: np.ndarray,
    values: np.ndarray,
    max_iter: int = 5000,
) -> dict:
    """
    Fit double-sigmoid curve to greenness time series.

    Uses scipy.optimize.curve_fit with bounds to fit the double-sigmoid
    model to observed GR values.

    Args:
        time: Time values (typically day of year), shape (N,).
        values: GR values corresponding to time points, shape (N,).
        max_iter: Maximum iterations for curve fitting. Default is 5000.

    Returns:
        Dictionary containing:
        - 'params': Fitted parameters (base, amp1, k1, t1, amp2, k2, t2)
        - 't1' (gup_doy): Green-up date (inflection of rising sigmoid)
        - 't2' (gdown_doy): Green-down date (inflection of falling sigmoid)
        - 'gmax_doy': Date of maximum fitted value
        - 'gmax_value': Maximum fitted value
        - 'fit_rmse': Root mean square error of fit
        - 'fit_status': 'success' or 'failed'
    """
    result = {
        "params": None,
        "t1": np.nan,
        "t2": np.nan,
        "gmax_doy": np.nan,
        "gmax_value": np.nan,
        "fit_rmse": np.nan,
        "fit_status": "failed",
    }

    # Need minimum number of points
    if len(time) < 30:
        logger.warning(f"Insufficient data points ({len(time)}) for double-sigmoid fit")
        return result

    # Remove NaN values
    valid_mask = ~np.isnan(values)
    t = time[valid_mask]
    v = values[valid_mask]

    if len(t) < 30:
        logger.warning(f"Insufficient valid data points ({len(t)}) after removing NaNs")
        return result

    # Estimate initial parameters
    v_min = np.min(v)
    v_max = np.max(v)
    v_range = v_max - v_min

    # Initial guesses
    base0 = v_min
    amp0 = v_range * 1.2  # Slightly larger to allow for overshoot
    k0 = 0.1  # Moderate rate
    t1_0 = t[np.argmax(np.gradient(v))] if len(v) > 2 else t.mean() - 60  # Green-up
    t2_0 = t[np.argmin(np.gradient(v))] if len(v) > 2 else t.mean() + 60  # Green-down

    p0 = [base0, amp0, k0, t1_0, amp0, k0, t2_0]

    # Parameter bounds
    t_min, t_max = t.min(), t.max()
    bounds = (
        [0, 0, 0.001, t_min, 0, 0.001, t_min],  # Lower bounds
        [1, 1, 0.5, t_max, 1, 0.5, t_max],  # Upper bounds
    )

    try:
        popt, _ = curve_fit(
            _double_sigmoid,
            t,
            v,
            p0=p0,
            bounds=bounds,
            maxfev=max_iter,
            method="trf",
        )

        # Compute fitted values and RMSE
        v_fitted = _double_sigmoid(t, *popt)
        rmse = np.sqrt(np.mean((v - v_fitted) ** 2))

        # Extract inflection points
        base, amp1, k1, t1, amp2, k2, t2 = popt

        # Find maximum of fitted curve
        t_fine = np.linspace(t_min, t_max, 365)
        v_fine = _double_sigmoid(t_fine, *popt)
        max_idx = np.argmax(v_fine)
        gmax_doy = t_fine[max_idx]
        gmax_value = v_fine[max_idx]

        result.update(
            {
                "params": popt,
                "t1": t1,  # Green-up DOY
                "t2": t2,  # Green-down DOY
                "gmax_doy": gmax_doy,
                "gmax_value": gmax_value,
                "fit_rmse": rmse,
                "fit_status": "success",
            }
        )

    except (RuntimeError, ValueError) as e:
        logger.warning(f"Curve fitting failed: {e}")

    return result


def phenology(
    gr_data: Union[dict, pd.DataFrame],
    model: str = "double_sigmoid",
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract phenology dates from GR time series.

    Fits a phenological model to the GR time series and extracts key dates:
    - gup_doy: Green-up date (inflection point of rising phase)
    - gdown_doy: Green-down date (inflection point of falling phase)
    - gmax_doy: Date of maximum greenness

    Args:
        gr_data: Either a DataFrame with 'doy' and 'mean_gr' columns,
            or a dict of date-indexed GR images from gr() function.
        model: Phenological model to use. Currently only 'double_sigmoid'
            is supported. Default is 'double_sigmoid'.
        output_dir: Optional directory to save fitted curve data.

    Returns:
        DataFrame with columns:
        - row, col: Pixel coordinates (if pixel-level fitting)
        - gup_doy: Green-up day of year
        - gdown_doy: Green-down day of year
        - gmax_doy: Day of maximum greenness
        - gmax_value: Maximum greenness value
        - fit_rmse: Root mean square error of fit
        - fit_status: 'success' or 'failed'

    Raises:
        ValueError: If model is not recognized or gr_data format is invalid.
    """
    if model != "double_sigmoid":
        raise ValueError(f"Unknown model: {model}. Only 'double_sigmoid' is supported.")

    # Handle DataFrame input
    if isinstance(gr_data, pd.DataFrame):
        # Check for required columns
        if "doy" not in gr_data.columns:
            if "date" in gr_data.columns:
                gr_data = gr_data.copy()
                gr_data["doy"] = pd.to_datetime(gr_data["date"]).dt.dayofyear
            else:
                raise ValueError("gr_data must have 'doy' or 'date' column")

        if "mean_gr" not in gr_data.columns:
            raise ValueError("gr_data must have 'mean_gr' column")

        # Fit to mean time series
        t = gr_data["doy"].values
        v = gr_data["mean_gr"].values

        fit_result = fit_double_sigmoid(t, v)

        result_df = pd.DataFrame(
            [
                {
                    "row": 0,
                    "col": 0,
                    "gup_doy": fit_result["t1"],
                    "gdown_doy": fit_result["t2"],
                    "gmax_doy": fit_result["gmax_doy"],
                    "gmax_value": fit_result["gmax_value"],
                    "fit_rmse": fit_result["fit_rmse"],
                    "fit_status": fit_result["fit_status"],
                }
            ]
        )

    elif isinstance(gr_data, dict):
        # Pixel-level fitting from dict of GR images
        # This is a more advanced use case for spatial phenology mapping
        raise NotImplementedError(
            "Pixel-level phenology from dict of images not yet implemented. "
            "Please provide a DataFrame with aggregated time series."
        )

    else:
        raise ValueError(f"gr_data must be DataFrame or dict, got {type(gr_data)}")

    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path / "phenology_results.csv", index=False)

    return result_df
