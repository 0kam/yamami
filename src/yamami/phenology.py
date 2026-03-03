"""
Vegetation phenology analysis module for YAMAMI.

This module provides functions for computing greenness ratio (GR) time series
and extracting phenological dates using double-sigmoid curve fitting or
threshold-based methods.

The main functions are:
- gr(): Compute greenness ratio from RGB images
- fit_double_sigmoid(): Fit double-sigmoid curve to time series
- phenology(): Extract phenological dates (green-up, green-max, green-down)

Example usage:
    >>> from yamami.phenology import gr, phenology
    >>> daily_images, timeseries = gr(profile_df, aoi_mask, aligned_dir="aligned/")
    >>> pheno_mean = phenology(timeseries)
    >>> pheno_pixel = phenology("gr_output/", timeseries=timeseries, aoi_mask=aoi_mask)
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

from yamami.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Double-sigmoid model
# =============================================================================


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

    Model: f(t) = base + amp1/(1+exp(-k1*(t-t1))) - amp2/(1+exp(-k2*(t-t2)))
    """
    rising = amp1 / (1 + np.exp(-k1 * (t - t1)))
    falling = amp2 / (1 + np.exp(-k2 * (t - t2)))
    return base + rising - falling


# =============================================================================
# GR computation
# =============================================================================


def _compute_gr_image(image: np.ndarray) -> np.ndarray:
    """
    Compute greenness ratio for a BGR image (OpenCV format).

    GR = G / (R + G + B)

    Args:
        image: BGR image with shape (H, W, 3), values in range [0, 255].

    Returns:
        Greenness ratio image with shape (H, W), dtype float32,
        values in range [0, 1].
    """
    img_float = image.astype(np.float32)

    b = img_float[:, :, 0]
    g = img_float[:, :, 1]
    r = img_float[:, :, 2]

    total = r + g + b

    with np.errstate(divide="ignore", invalid="ignore"):
        gr_img = np.where(total > 0, g / total, np.float32(0.0))

    return gr_img.astype(np.float32)


def _resolve_image_path(
    original_path: str, aligned_dir: Optional[str]
) -> str:
    """Resolve image path, preferring aligned version if available."""
    if aligned_dir is None:
        return str(original_path)

    filename = Path(original_path).name
    aligned_path = Path(aligned_dir) / filename
    if aligned_path.exists():
        return str(aligned_path)

    return str(original_path)


def gr(
    profile: pd.DataFrame,
    aoi_mask: np.ndarray,
    daily_max: bool = True,
    output_dir: Optional[str] = None,
    aligned_dir: Optional[str] = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Compute greenness ratio time series from image profile.

    Calculates GR = G / (R + G + B) for each pixel within the AOI mask.
    When ``daily_max=True``, computes pixel-level daily maximum GR
    (taking the per-pixel max across all images on the same date).

    Args:
        profile: DataFrame with 'timestamp' column and 'path' or 'filepath'
            column pointing to original image locations.
        aoi_mask: Boolean mask array (H, W) defining the area of interest.
        daily_max: If True, aggregate to per-pixel daily maximum. Default True.
        output_dir: Directory to save daily GR images as .npy files.
            When provided, images are saved to disk and NOT kept in the
            returned dict (to conserve memory for large datasets).
        aligned_dir: Directory containing aligned images. When provided,
            images are loaded from this directory (matched by filename)
            instead of the original paths.

    Returns:
        Tuple of (daily_images, timeseries):
        - daily_images: Dict mapping date strings ("YYYY-MM-DD") to
          GR image arrays (float32, shape H×W). Empty when output_dir
          is provided (images are saved to disk instead).
        - timeseries: DataFrame with columns [date, doy, mean_gr, n_images].

    Raises:
        ValueError: If profile is empty or missing required columns.
    """
    if profile.empty:
        logger.warning("Empty profile DataFrame provided")
        return {}, pd.DataFrame(columns=["date", "doy", "mean_gr"])

    # Normalize column names
    df = profile.copy()
    if "filepath" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "filepath"})
    if "filepath" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("Profile must have 'path'/'filepath' and 'timestamp' columns")

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Prepare output directory
    out_path = None
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    daily_images: dict[str, np.ndarray] = {}
    timeseries_records: list[dict] = []
    save_to_disk = output_dir is not None

    if daily_max:
        # Group by date and compute pixel-level daily max
        grouped = df.groupby("date")
        for date, group in tqdm(grouped, desc="Computing daily GR"):
            date_str = str(date)
            max_gr: Optional[np.ndarray] = None
            n_valid = 0

            for _, row in group.iterrows():
                img_path = _resolve_image_path(row["filepath"], aligned_dir)
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Could not load image: {img_path}")
                    continue

                gr_img = _compute_gr_image(img)

                if aoi_mask.shape != gr_img.shape:
                    logger.warning(
                        f"Mask shape {aoi_mask.shape} != image shape "
                        f"{gr_img.shape}, skipping {img_path}"
                    )
                    continue

                if max_gr is None:
                    max_gr = gr_img.copy()
                else:
                    np.maximum(max_gr, gr_img, out=max_gr)
                n_valid += 1

            if max_gr is None:
                continue

            aoi_values = max_gr[aoi_mask]
            if len(aoi_values) == 0:
                continue
            mean_gr = float(np.nanmean(aoi_values))

            if save_to_disk:
                np.save(out_path / f"gr_{date_str}.npy", max_gr)
            else:
                daily_images[date_str] = max_gr

            timeseries_records.append({
                "date": date,
                "doy": pd.Timestamp(date).dayofyear,
                "mean_gr": mean_gr,
                "n_images": n_valid,
            })
    else:
        for _, row in tqdm(df.iterrows(), desc="Computing GR", total=len(df)):
            img_path = _resolve_image_path(row["filepath"], aligned_dir)
            timestamp = row["timestamp"]
            date = row["date"]

            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not load image: {img_path}")
                continue

            gr_img = _compute_gr_image(img)
            if aoi_mask.shape != gr_img.shape:
                continue

            aoi_values = gr_img[aoi_mask]
            if len(aoi_values) == 0:
                continue
            mean_gr = float(np.nanmean(aoi_values))

            key = f"{date}_{timestamp.strftime('%H%M')}"

            if save_to_disk:
                np.save(out_path / f"gr_{key}.npy", gr_img)
            else:
                daily_images[key] = gr_img

            timeseries_records.append({
                "date": date,
                "doy": pd.Timestamp(date).dayofyear,
                "mean_gr": mean_gr,
                "n_images": 1,
            })

    if not timeseries_records:
        logger.warning("No valid GR values computed")
        return {}, pd.DataFrame(columns=["date", "doy", "mean_gr"])

    timeseries = pd.DataFrame(timeseries_records)
    timeseries["date"] = pd.to_datetime(timeseries["date"])
    timeseries = timeseries.sort_values("doy").reset_index(drop=True)

    if output_dir:
        timeseries.to_csv(out_path / "gr_timeseries.csv", index=False)

    logger.info(
        f"GR computation complete: {len(timeseries)} dates, "
        f"DOY range {timeseries['doy'].min()}-{timeseries['doy'].max()}"
    )

    return daily_images, timeseries


# =============================================================================
# Double-sigmoid fitting
# =============================================================================


def fit_double_sigmoid(
    time: np.ndarray,
    values: np.ndarray,
    max_iter: int = 5000,
    p0: Optional[np.ndarray] = None,
) -> dict:
    """
    Fit double-sigmoid curve to greenness time series.

    Args:
        time: Time values (typically day of year), shape (N,).
        values: GR values corresponding to time points, shape (N,).
        max_iter: Maximum iterations for curve fitting.
        p0: Optional initial parameter guess [base, amp1, k1, t1, amp2, k2, t2].

    Returns:
        Dictionary with keys: params, t1, t2, gmax_doy, gmax_value,
        fit_rmse, fit_status.
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

    if len(time) < 10:
        return result

    valid_mask = ~(np.isnan(values) | np.isnan(time))
    t = time[valid_mask].astype(np.float64)
    v = values[valid_mask].astype(np.float64)

    if len(t) < 10:
        return result

    t_min, t_max = float(t.min()), float(t.max())
    v_min, v_max = float(np.min(v)), float(np.max(v))
    v_range = v_max - v_min

    if v_range < 1e-6:
        return result

    if p0 is None:
        base0 = v_min
        amp0 = v_range * 1.2

        if len(v) > 5:
            grad = np.gradient(v, t)
            kernel = np.ones(min(5, len(grad))) / min(5, len(grad))
            grad_smooth = np.convolve(grad, kernel, mode="same")
            t1_0 = float(t[np.argmax(grad_smooth)])
            t2_0 = float(t[np.argmin(grad_smooth)])
            if t1_0 > t2_0:
                t1_0, t2_0 = t2_0, t1_0
        else:
            t_mid = (t_min + t_max) / 2
            t1_0 = t_mid - (t_max - t_min) * 0.2
            t2_0 = t_mid + (t_max - t_min) * 0.2

        p0 = np.array([base0, amp0, 0.1, t1_0, amp0, 0.1, t2_0])

    lower = np.array([0.0, 0.0, 0.001, t_min, 0.0, 0.001, t_min])
    upper = np.array([1.0, 1.0, 1.0, t_max, 1.0, 1.0, t_max])
    p0_clamped = np.clip(p0, lower + 1e-6, upper - 1e-6)

    try:
        popt, _ = curve_fit(
            _double_sigmoid, t, v,
            p0=p0_clamped, bounds=(lower, upper),
            maxfev=max_iter, method="trf",
        )

        v_fitted = _double_sigmoid(t, *popt)
        rmse = float(np.sqrt(np.mean((v - v_fitted) ** 2)))

        base, amp1, k1, t1, amp2, k2, t2 = popt

        t_fine = np.linspace(t_min, t_max, 1000)
        v_fine = _double_sigmoid(t_fine, *popt)
        max_idx = int(np.argmax(v_fine))

        result.update({
            "params": popt,
            "t1": float(t1),
            "t2": float(t2),
            "gmax_doy": float(t_fine[max_idx]),
            "gmax_value": float(v_fine[max_idx]),
            "fit_rmse": rmse,
            "fit_status": "success",
        })

    except (RuntimeError, ValueError, TypeError) as e:
        logger.debug(f"Curve fitting failed: {e}")

    return result


# =============================================================================
# Vectorized threshold-based pixel phenology
# =============================================================================


def _load_gr_strip(
    gr_dir: Path,
    date_strings: list[str],
    row_slice: slice,
) -> np.ndarray:
    """
    Load a full-width row strip from saved daily GR .npy files.

    Returns:
        Array of shape (T, strip_h, W), dtype float32.
    """
    n_dates = len(date_strings)

    strip_shape = None
    for ds in date_strings:
        npy_path = gr_dir / f"gr_{ds}.npy"
        if npy_path.exists():
            full = np.load(npy_path, mmap_mode="r")
            strip_shape = full[row_slice, :].shape
            break

    if strip_shape is None:
        raise ValueError("No GR .npy files found in directory")

    strip = np.full((n_dates, *strip_shape), np.nan, dtype=np.float32)

    for t_idx, ds in enumerate(date_strings):
        npy_path = gr_dir / f"gr_{ds}.npy"
        if not npy_path.exists():
            continue
        full = np.load(npy_path, mmap_mode="r")
        strip[t_idx] = full[row_slice, :]

    return strip


def _threshold_phenology_strip(
    strip: np.ndarray,
    doy: np.ndarray,
    mask: np.ndarray,
    smooth_window: int = 7,
    gup_ratio: float = 0.5,
    gdown_ratio: float = 0.5,
    min_range: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute threshold-based phenology for a row strip.

    Uses a smoothed GR time series to find:
    - gup_doy: First DOY where smoothed GR rises above threshold
    - gdown_doy: Last DOY where smoothed GR is above threshold
    - gmax_doy: DOY of maximum smoothed GR
    - gmax_value: Maximum smoothed GR value

    The threshold is defined as: base + ratio * amplitude, where
    base = seasonal minimum and amplitude = max - min.

    Args:
        strip: GR values, shape (T, strip_h, W).
        doy: Day-of-year array, shape (T,).
        mask: AOI mask, shape (strip_h, W).
        smooth_window: Temporal smoothing window size.
        gup_ratio: Threshold ratio for green-up detection.
        gdown_ratio: Threshold ratio for green-down detection.
        min_range: Minimum GR range below which pixel is "low_variation".

    Returns:
        Tuple of (gup_map, gdown_map, gmax_map, gmax_val_map),
        each of shape (strip_h, W), float32. Non-AOI pixels are NaN.
    """
    T, strip_h, strip_w = strip.shape
    nan_val = np.float32(np.nan)

    gup_map = np.full((strip_h, strip_w), nan_val, dtype=np.float32)
    gdown_map = np.full((strip_h, strip_w), nan_val, dtype=np.float32)
    gmax_map = np.full((strip_h, strip_w), nan_val, dtype=np.float32)
    gmax_val_map = np.full((strip_h, strip_w), nan_val, dtype=np.float32)

    if not mask.any():
        return gup_map, gdown_map, gmax_map, gmax_val_map

    # Smooth along time axis (simple moving average, NaN-aware)
    if smooth_window > 1 and T >= smooth_window:
        from scipy.ndimage import uniform_filter1d
        # uniform_filter1d handles edges with mirror mode
        smooth = uniform_filter1d(
            strip.astype(np.float32), size=smooth_window, axis=0
        )
    else:
        smooth = strip.copy()

    # Seasonal range per pixel
    v_min = np.nanmin(smooth, axis=0)  # (strip_h, W)
    v_max = np.nanmax(smooth, axis=0)  # (strip_h, W)
    v_range = v_max - v_min

    # Skip low-variation pixels
    active = mask & (v_range >= min_range)
    if not active.any():
        return gup_map, gdown_map, gmax_map, gmax_val_map

    # Thresholds
    thresh_up = v_min + gup_ratio * v_range
    thresh_down = v_min + gdown_ratio * v_range

    # Green-max: DOY of maximum smoothed GR
    max_idx = np.nanargmax(smooth, axis=0)  # (strip_h, W)
    gmax_map[active] = doy[max_idx[active]].astype(np.float32)
    gmax_val_map[active] = v_max[active]

    # Green-up: first DOY where smooth >= threshold
    above_up = smooth >= thresh_up[np.newaxis, :, :]  # (T, strip_h, W)
    # Mask non-active pixels to avoid false positives
    above_up[:, ~active] = False

    # argmax on bool finds first True; returns 0 if none True
    first_idx = np.argmax(above_up, axis=0)  # (strip_h, W)
    any_above = above_up.any(axis=0)

    gup_map[active & any_above] = doy[
        first_idx[active & any_above]
    ].astype(np.float32)

    # Green-down: last DOY where smooth >= threshold
    above_down = smooth >= thresh_down[np.newaxis, :, :]
    above_down[:, ~active] = False

    # Find last True by reversing time axis
    last_idx = T - 1 - np.argmax(above_down[::-1], axis=0)
    any_above_down = above_down.any(axis=0)

    gdown_map[active & any_above_down] = doy[
        last_idx[active & any_above_down]
    ].astype(np.float32)

    return gup_map, gdown_map, gmax_map, gmax_val_map


def _phenology_threshold_from_dir(
    gr_dir: str,
    timeseries: pd.DataFrame,
    aoi_mask: np.ndarray,
    strip_height: int,
    smooth_window: int,
    gup_ratio: float,
    gdown_ratio: float,
    min_range: float,
    output_dir: Optional[str],
) -> pd.DataFrame:
    """Pixel-level phenology using threshold method from GR directory."""
    gr_path = Path(gr_dir)

    ts = timeseries.sort_values("doy").reset_index(drop=True)
    date_strings = [str(d.date()) for d in pd.to_datetime(ts["date"])]
    doy = ts["doy"].values.astype(np.float32)

    h, w = aoi_mask.shape
    n_strips = (h + strip_height - 1) // strip_height

    # Initialize full raster maps
    gup_raster = np.full((h, w), np.nan, dtype=np.float32)
    gdown_raster = np.full((h, w), np.nan, dtype=np.float32)
    gmax_raster = np.full((h, w), np.nan, dtype=np.float32)
    gmax_val_raster = np.full((h, w), np.nan, dtype=np.float32)

    logger.info(
        f"Threshold phenology: {h}x{w} image, {n_strips} strips of "
        f"{strip_height} rows, {len(doy)} dates, "
        f"{int(aoi_mask.sum()):,} AOI pixels"
    )

    for si in tqdm(range(n_strips), desc="Computing phenology"):
        r0 = si * strip_height
        r1 = min(r0 + strip_height, h)
        row_slice = slice(r0, r1)

        strip_mask = aoi_mask[row_slice, :]
        if not strip_mask.any():
            continue

        # Load strip cube (T, strip_h, W)
        strip_cube = _load_gr_strip(gr_path, date_strings, row_slice)

        # Compute threshold phenology for this strip
        gup, gdown, gmax, gmax_val = _threshold_phenology_strip(
            strip_cube, doy, strip_mask,
            smooth_window=smooth_window,
            gup_ratio=gup_ratio,
            gdown_ratio=gdown_ratio,
            min_range=min_range,
        )

        gup_raster[row_slice, :] = gup
        gdown_raster[row_slice, :] = gdown
        gmax_raster[row_slice, :] = gmax
        gmax_val_raster[row_slice, :] = gmax_val

        del strip_cube

    # Convert rasters to DataFrame (only AOI pixels)
    rows, cols = np.where(aoi_mask)
    result_df = pd.DataFrame({
        "row": rows,
        "col": cols,
        "gup_doy": gup_raster[rows, cols],
        "gdown_doy": gdown_raster[rows, cols],
        "gmax_doy": gmax_raster[rows, cols],
        "gmax_value": gmax_val_raster[rows, cols],
        "fit_rmse": np.float32(np.nan),
        "fit_status": "threshold",
    })

    # Mark low_variation and insufficient_data
    nan_mask = np.isnan(result_df["gup_doy"])
    low_var = nan_mask & (
        (gmax_val_raster[rows, cols] - np.nanmin(gup_raster)) < min_range
    )
    result_df.loc[nan_mask, "fit_status"] = "low_variation"

    if output_dir:
        _save_raster_maps(
            gup_raster, gdown_raster, gmax_raster, gmax_val_raster,
            result_df, output_dir,
        )

    n_valid = (~np.isnan(result_df["gup_doy"])).sum()
    logger.info(
        f"Threshold phenology: {n_valid}/{len(result_df)} pixels "
        f"with valid green-up dates"
    )

    return result_df


def _save_raster_maps(
    gup: np.ndarray,
    gdown: np.ndarray,
    gmax: np.ndarray,
    gmax_val: np.ndarray,
    result_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save phenology raster maps as npy files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "gup_doy.npy", gup)
    np.save(out / "gdown_doy.npy", gdown)
    np.save(out / "gmax_doy.npy", gmax)
    np.save(out / "gmax_value.npy", gmax_val)

    result_df.to_csv(out / "phenology_results.csv", index=False)

    logger.info(f"Saved phenology maps to {out}")


# =============================================================================
# Public phenology API
# =============================================================================


def phenology(
    gr_data: Union[str, dict, pd.DataFrame],
    timeseries: Optional[pd.DataFrame] = None,
    aoi_mask: Optional[np.ndarray] = None,
    method: str = "threshold",
    output_dir: Optional[str] = None,
    block_size: int = 128,
    smooth_window: int = 7,
    gup_ratio: float = 0.5,
    gdown_ratio: float = 0.5,
    min_range: float = 0.02,
) -> pd.DataFrame:
    """
    Extract phenology dates from GR time series.

    Supports three input modes:

    1. **Mean time series** (``gr_data`` is DataFrame):
       Fits a double-sigmoid to the mean GR time series (ignores ``method``).
       Returns one-row DataFrame with aggregated phenology dates.

    2. **Pixel-level from directory** (``gr_data`` is str path):
       Reads daily GR .npy files saved by ``gr()``. Requires ``timeseries``
       and ``aoi_mask``.

    3. **Pixel-level from dict** (``gr_data`` is dict):
       Uses in-memory daily GR images. Requires ``aoi_mask``.

    Args:
        gr_data: One of:
            - DataFrame with 'doy' and 'mean_gr' columns (mean fitting)
            - str path to directory with gr_*.npy files (pixel fitting)
            - dict mapping date strings to GR arrays (pixel fitting)
        timeseries: DataFrame with 'date' and 'doy' columns.
            Required when gr_data is a directory path.
        aoi_mask: Boolean mask (H, W). Required for pixel-level fitting.
        method: Pixel-level method. ``"threshold"`` (default) uses fast
            vectorized threshold crossings. ``"double_sigmoid"`` uses
            per-pixel curve fitting (very slow for large images).
            Ignored when gr_data is a DataFrame (always uses double_sigmoid).
        output_dir: Directory to save phenology maps and CSV results.
        block_size: Spatial block/strip height for pixel-level processing.
        smooth_window: Temporal smoothing window (threshold method only).
        gup_ratio: Green-up threshold as fraction of seasonal amplitude.
        gdown_ratio: Green-down threshold as fraction of seasonal amplitude.
        min_range: Minimum GR range (max-min) to consider a pixel as having
            seasonal variation. Pixels below this are marked "low_variation".

    Returns:
        DataFrame with columns: row, col, gup_doy, gdown_doy,
        gmax_doy, gmax_value, fit_rmse, fit_status.
    """
    # --- Mode 1: Mean time series (always uses double_sigmoid) ---
    if isinstance(gr_data, pd.DataFrame):
        if "doy" not in gr_data.columns:
            if "date" in gr_data.columns:
                gr_data = gr_data.copy()
                gr_data["doy"] = pd.to_datetime(gr_data["date"]).dt.dayofyear
            else:
                raise ValueError("gr_data must have 'doy' or 'date' column")

        if "mean_gr" not in gr_data.columns:
            raise ValueError("gr_data must have 'mean_gr' column")

        t = gr_data["doy"].values
        v = gr_data["mean_gr"].values

        fit_result = fit_double_sigmoid(t, v)

        result_df = pd.DataFrame([{
            "row": 0,
            "col": 0,
            "gup_doy": fit_result["t1"],
            "gdown_doy": fit_result["t2"],
            "gmax_doy": fit_result["gmax_doy"],
            "gmax_value": fit_result["gmax_value"],
            "fit_rmse": fit_result["fit_rmse"],
            "fit_status": fit_result["fit_status"],
        }])

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(out / "phenology_results.csv", index=False)

            if fit_result["params"] is not None:
                t_fine = np.linspace(t.min(), t.max(), 365)
                v_fine = _double_sigmoid(t_fine, *fit_result["params"])
                curve_df = pd.DataFrame({"doy": t_fine, "fitted_gr": v_fine})
                curve_df.to_csv(out / "fitted_curve.csv", index=False)

        return result_df

    # --- Pixel-level modes ---
    if isinstance(gr_data, str):
        if timeseries is None:
            raise ValueError(
                "timeseries DataFrame is required when gr_data is a directory path"
            )
        if aoi_mask is None:
            raise ValueError(
                "aoi_mask is required for pixel-level phenology"
            )

        if method == "threshold":
            return _phenology_threshold_from_dir(
                gr_data, timeseries, aoi_mask, block_size,
                smooth_window, gup_ratio, gdown_ratio, min_range,
                output_dir,
            )
        elif method == "double_sigmoid":
            # Get p0 hint from mean fit
            p0_hint = _get_p0_hint(timeseries)
            return _phenology_sigmoid_from_dir(
                gr_data, timeseries, aoi_mask, p0_hint,
                block_size, output_dir,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    if isinstance(gr_data, dict):
        if aoi_mask is None:
            raise ValueError(
                "aoi_mask is required for pixel-level phenology"
            )

        if method == "threshold":
            return _phenology_threshold_from_dict(
                gr_data, aoi_mask, block_size,
                smooth_window, gup_ratio, gdown_ratio, min_range,
                output_dir,
            )
        elif method == "double_sigmoid":
            p0_hint = _get_p0_hint(timeseries)
            return _phenology_sigmoid_from_dict(
                gr_data, aoi_mask, p0_hint, block_size, output_dir,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    raise ValueError(f"gr_data must be DataFrame, str, or dict, got {type(gr_data)}")


# =============================================================================
# Helper: p0 hint from mean timeseries
# =============================================================================

def _get_p0_hint(timeseries: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
    """Extract p0 hint from mean timeseries fit."""
    if timeseries is None or len(timeseries) < 10:
        return None

    ts = timeseries.copy()
    if "doy" not in ts.columns and "date" in ts.columns:
        ts["doy"] = pd.to_datetime(ts["date"]).dt.dayofyear
    if "mean_gr" not in ts.columns:
        return None

    mean_fit = fit_double_sigmoid(ts["doy"].values, ts["mean_gr"].values)
    if mean_fit["params"] is not None:
        logger.info(
            f"Using mean fit as initial guess: "
            f"gup={mean_fit['t1']:.0f}, gdown={mean_fit['t2']:.0f}"
        )
        return mean_fit["params"]
    return None


# =============================================================================
# Double-sigmoid per-pixel (slow, for small regions)
# =============================================================================


def _fit_pixel_block(
    doy: np.ndarray,
    block: np.ndarray,
    mask_block: np.ndarray,
    p0_hint: Optional[np.ndarray],
    row_offset: int,
    col_offset: int,
    min_valid: int = 20,
    min_range: float = 0.02,
) -> list[dict]:
    """Fit double-sigmoid to each pixel in a spatial block."""
    block_h, block_w = mask_block.shape
    results = []

    for i in range(block_h):
        for j in range(block_w):
            if not mask_block[i, j]:
                continue

            pixel_ts = block[:, i, j]
            valid = ~np.isnan(pixel_ts)
            n_valid = int(valid.sum())

            row = row_offset + i
            col = col_offset + j

            if n_valid < min_valid:
                results.append({
                    "row": row, "col": col,
                    "gup_doy": np.nan, "gdown_doy": np.nan,
                    "gmax_doy": np.nan, "gmax_value": np.nan,
                    "fit_rmse": np.nan, "fit_status": "insufficient_data",
                })
                continue

            v = pixel_ts[valid]
            if float(np.max(v) - np.min(v)) < min_range:
                results.append({
                    "row": row, "col": col,
                    "gup_doy": np.nan, "gdown_doy": np.nan,
                    "gmax_doy": np.nan, "gmax_value": np.nan,
                    "fit_rmse": np.nan, "fit_status": "low_variation",
                })
                continue

            fit = fit_double_sigmoid(doy[valid], v, max_iter=3000, p0=p0_hint)
            results.append({
                "row": row, "col": col,
                "gup_doy": fit["t1"], "gdown_doy": fit["t2"],
                "gmax_doy": fit["gmax_doy"], "gmax_value": fit["gmax_value"],
                "fit_rmse": fit["fit_rmse"], "fit_status": fit["fit_status"],
            })

    return results


def _phenology_sigmoid_from_dir(
    gr_dir: str,
    timeseries: pd.DataFrame,
    aoi_mask: np.ndarray,
    p0_hint: Optional[np.ndarray],
    block_size: int,
    output_dir: Optional[str],
) -> pd.DataFrame:
    """Pixel-level double-sigmoid phenology from GR directory."""
    gr_path = Path(gr_dir)

    ts = timeseries.sort_values("doy").reset_index(drop=True)
    date_strings = [str(d.date()) for d in pd.to_datetime(ts["date"])]
    doy = ts["doy"].values.astype(np.float64)

    h, w = aoi_mask.shape
    all_results: list[dict] = []

    n_row_strips = (h + block_size - 1) // block_size
    n_col_blocks = (w + block_size - 1) // block_size
    n_aoi_pixels = int(aoi_mask.sum())

    logger.info(
        f"Sigmoid pixel phenology: {h}x{w}, block={block_size}, "
        f"{len(doy)} dates, {n_aoi_pixels:,} AOI pixels"
    )

    with tqdm(total=n_aoi_pixels, desc="Fitting phenology", unit="px") as pbar:
        for rb in range(n_row_strips):
            r0 = rb * block_size
            r1 = min(r0 + block_size, h)
            row_slice = slice(r0, r1)

            strip_mask = aoi_mask[row_slice, :]
            if not strip_mask.any():
                continue

            strip_cube = _load_gr_strip(gr_path, date_strings, row_slice)

            for cb in range(n_col_blocks):
                c0 = cb * block_size
                c1 = min(c0 + block_size, w)
                col_slice = slice(c0, c1)

                mask_block = aoi_mask[row_slice, col_slice]
                if not mask_block.any():
                    continue

                block_results = _fit_pixel_block(
                    doy, strip_cube[:, :, col_slice], mask_block,
                    p0_hint, row_offset=r0, col_offset=c0,
                )
                all_results.extend(block_results)
                pbar.update(int(mask_block.sum()))

            del strip_cube

    if not all_results:
        return pd.DataFrame(columns=[
            "row", "col", "gup_doy", "gdown_doy",
            "gmax_doy", "gmax_value", "fit_rmse", "fit_status",
        ])

    result_df = pd.DataFrame(all_results)

    if output_dir:
        _save_raster_from_df(result_df, aoi_mask.shape, output_dir)

    n_success = (result_df["fit_status"] == "success").sum()
    logger.info(f"Sigmoid phenology: {n_success}/{len(result_df)} fitted")

    return result_df


def _phenology_threshold_from_dict(
    daily_images: dict[str, np.ndarray],
    aoi_mask: np.ndarray,
    strip_height: int,
    smooth_window: int,
    gup_ratio: float,
    gdown_ratio: float,
    min_range: float,
    output_dir: Optional[str],
) -> pd.DataFrame:
    """Threshold phenology from in-memory dict of GR images."""
    sorted_dates = sorted(daily_images.keys())
    doy = np.array([
        pd.Timestamp(d).dayofyear for d in sorted_dates
    ], dtype=np.float32)

    h, w = aoi_mask.shape
    n_strips = (h + strip_height - 1) // strip_height

    gup_raster = np.full((h, w), np.nan, dtype=np.float32)
    gdown_raster = np.full((h, w), np.nan, dtype=np.float32)
    gmax_raster = np.full((h, w), np.nan, dtype=np.float32)
    gmax_val_raster = np.full((h, w), np.nan, dtype=np.float32)

    for si in tqdm(range(n_strips), desc="Computing phenology"):
        r0 = si * strip_height
        r1 = min(r0 + strip_height, h)
        row_slice = slice(r0, r1)

        strip_mask = aoi_mask[row_slice, :]
        if not strip_mask.any():
            continue

        strip_cube = np.stack(
            [daily_images[ds][row_slice, :] for ds in sorted_dates],
            axis=0,
        )

        gup, gdown, gmax, gmax_val = _threshold_phenology_strip(
            strip_cube, doy, strip_mask,
            smooth_window=smooth_window,
            gup_ratio=gup_ratio,
            gdown_ratio=gdown_ratio,
            min_range=min_range,
        )

        gup_raster[row_slice, :] = gup
        gdown_raster[row_slice, :] = gdown
        gmax_raster[row_slice, :] = gmax
        gmax_val_raster[row_slice, :] = gmax_val

    rows, cols = np.where(aoi_mask)
    result_df = pd.DataFrame({
        "row": rows,
        "col": cols,
        "gup_doy": gup_raster[rows, cols],
        "gdown_doy": gdown_raster[rows, cols],
        "gmax_doy": gmax_raster[rows, cols],
        "gmax_value": gmax_val_raster[rows, cols],
        "fit_rmse": np.float32(np.nan),
        "fit_status": "threshold",
    })

    nan_mask = np.isnan(result_df["gup_doy"])
    result_df.loc[nan_mask, "fit_status"] = "low_variation"

    if output_dir:
        _save_raster_maps(
            gup_raster, gdown_raster, gmax_raster, gmax_val_raster,
            result_df, output_dir,
        )

    n_valid = (~np.isnan(result_df["gup_doy"])).sum()
    logger.info(f"Threshold phenology: {n_valid}/{len(result_df)} valid")

    return result_df


def _phenology_sigmoid_from_dict(
    daily_images: dict[str, np.ndarray],
    aoi_mask: np.ndarray,
    p0_hint: Optional[np.ndarray],
    block_size: int,
    output_dir: Optional[str],
) -> pd.DataFrame:
    """Double-sigmoid phenology from in-memory dict."""
    sorted_dates = sorted(daily_images.keys())
    doy = np.array([
        pd.Timestamp(d).dayofyear for d in sorted_dates
    ], dtype=np.float64)

    h, w = aoi_mask.shape
    all_results: list[dict] = []

    n_row_blocks = (h + block_size - 1) // block_size
    n_col_blocks = (w + block_size - 1) // block_size
    n_aoi = int(aoi_mask.sum())

    with tqdm(total=n_aoi, desc="Fitting phenology", unit="px") as pbar:
        for rb in range(n_row_blocks):
            r0 = rb * block_size
            r1 = min(r0 + block_size, h)
            row_slice = slice(r0, r1)

            for cb in range(n_col_blocks):
                c0 = cb * block_size
                c1 = min(c0 + block_size, w)
                col_slice = slice(c0, c1)

                mask_block = aoi_mask[row_slice, col_slice]
                if not mask_block.any():
                    continue

                cube = np.stack(
                    [daily_images[ds][row_slice, col_slice] for ds in sorted_dates],
                    axis=0,
                ).astype(np.float32)

                block_results = _fit_pixel_block(
                    doy, cube, mask_block, p0_hint,
                    row_offset=r0, col_offset=c0,
                )
                all_results.extend(block_results)
                pbar.update(int(mask_block.sum()))

    if not all_results:
        return pd.DataFrame(columns=[
            "row", "col", "gup_doy", "gdown_doy",
            "gmax_doy", "gmax_value", "fit_rmse", "fit_status",
        ])

    result_df = pd.DataFrame(all_results)

    if output_dir:
        _save_raster_from_df(result_df, aoi_mask.shape, output_dir)

    n_success = (result_df["fit_status"] == "success").sum()
    logger.info(f"Sigmoid phenology: {n_success}/{len(result_df)} fitted")

    return result_df


def _save_raster_from_df(
    result_df: pd.DataFrame, img_shape: tuple, output_dir: str
) -> None:
    """Save per-pixel phenology results as rasters from DataFrame."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    h, w = img_shape
    for col_name in ["gup_doy", "gdown_doy", "gmax_doy", "gmax_value", "fit_rmse"]:
        raster = np.full((h, w), np.nan, dtype=np.float32)
        valid = result_df[col_name].notna()
        if valid.any():
            rows = result_df.loc[valid, "row"].values
            cols = result_df.loc[valid, "col"].values
            raster[rows, cols] = result_df.loc[valid, col_name].values.astype(np.float32)
        np.save(out / f"{col_name}.npy", raster)

    result_df.to_csv(out / "phenology_results.csv", index=False)
    logger.info(f"Saved phenology maps to {out}")
