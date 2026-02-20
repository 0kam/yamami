"""
Snow detection and snowmelt date estimation module.

Snow detection uses a BDN (Blue Digital Number) threshold method:
1. For each image with SAM snow/mountain labels, find the Blue channel
   threshold that maximises F1 between SAM snow (positive) and SAM
   mountain-non-snow (negative) histograms.
2. Images without SAM snow labels inherit the threshold from the
   learned image with the most similar AOI blue-channel histogram
   (falls back to temporal nearest if histogram unavailable).
3. Classify pixels as snow where Blue >= threshold within the valid mask.

Snowmelt estimation uses a two-stage approach:
  Stage 1: Image-level reliability weighting (threshold source quality,
           isotonic regression residual of image snow fraction).
  Stage 2: Per-pixel weighted Bernoulli change-point detection with
           daily aggregation.

SAM masks are warped to aligned-image coordinates using the alignment
homography before use.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from yamami.logging import get_logger

logger = get_logger(__name__)

# SAM labels whose pixels are excluded from classification (snow → NA)
_EXCLUDE_PROMPTS = frozenset([
    "sky", "cloud", "fog", "sun", "reflection",
    "human", "raindrop", "vehicle",
])

# BDN threshold search range
_MIN_BDN = 127
_MAX_BDN = 255


# =========================================================================
# Public API
# =========================================================================


def snow(
    df: pd.DataFrame,
    aligned_dir: str,
    masks_dir: str,
    mask: np.ndarray,
    fallback_threshold: int = 230,
    output_dir: Optional[str] = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Snow detection using BDN (Blue Digital Number) thresholding.

    Per-image Blue channel threshold is learned from SAM snow vs mountain
    histograms (F1 maximisation).  Images without SAM snow inherit the
    threshold from the learned image with the most similar AOI blue-channel
    histogram (histogram correlation matching).

    Args:
        df: DataFrame with ``filename``, ``timestamp``, ``segment_id``,
            ``align_status`` columns (from the alignment step).
        aligned_dir: Directory containing aligned images and
            ``alignment_params.json``.
        masks_dir: Directory containing SAM mask ``.npz`` files
            (in original, unwarped coordinates).
        mask: Boolean mask defining the region of interest
            (aligned-image coordinates).  Used both for threshold
            learning and snow classification.
        fallback_threshold: Constant threshold used when no learned
            threshold is available at all (default 230).
        output_dir: If given, save snow masks (.npz), per-image
            statistics (CSV), and side-by-side preview images.

    Returns:
        ``(snow_masks, stats_df)``

        *snow_masks*: ``{filename: bool ndarray}`` — per-pixel snow
        classification.  Excluded (NA) pixels are ``False``; use
        the ``valid_mask`` from ``_build_region_masks`` to distinguish
        NA from true negatives.

        *stats_df*: DataFrame with per-image snow statistics including
        the threshold used, its source, and classification counts.
    """
    from tqdm import tqdm

    aligned_dir = Path(aligned_dir)
    masks_dir = Path(masks_dir)

    # Load alignment parameters for mask warping
    params_path = aligned_dir / "alignment_params.json"
    with open(params_path) as f:
        align_params = json.load(f)

    # Filter to successfully aligned images, sort by timestamp
    work_df = df[df["align_status"] == "success"].copy()
    work_df = work_df.sort_values("timestamp").reset_index(drop=True)
    n_images = len(work_df)

    if n_images == 0:
        logger.warning("No aligned images to process")
        return {}, pd.DataFrame()

    # Build per-segment distortion remap tables
    img_h, img_w = mask.shape[:2]
    seg_remap = _build_segment_remap_tables(align_params, img_h, img_w)

    # ---- Pass 1: Learn per-image BDN thresholds ----
    logger.info(f"Pass 1: Learning BDN thresholds from {n_images} images")
    pass1_records: list[dict] = []
    learned_sources: list[dict] = []
    all_histograms: list[Optional[np.ndarray]] = [None] * n_images
    mask_u8 = (mask > 0).astype(np.uint8)

    for i in tqdm(range(n_images), desc="snow:threshold"):
        row = work_df.iloc[i]
        filename = row["filename"]
        basename = Path(filename).stem
        timestamp = pd.Timestamp(row["timestamp"])

        img = cv2.imread(str(aligned_dir / filename))
        if img is None:
            pass1_records.append({
                "index": i, "filename": filename,
                "timestamp": timestamp, "status": "skip_no_image",
                "threshold": None, "fit_prec": None,
                "fit_rec": None, "fit_f1": None,
                "source": None, "source_basename": None,
                "fit_pos": 0, "fit_neg": 0,
            })
            continue

        # Compute AOI blue histogram for threshold matching
        blue = img[:, :, 0]
        aoi_hist = cv2.calcHist(
            [blue], [0], mask_u8, [256], [0, 256],
        ).flatten()
        aoi_hist = aoi_hist / (aoi_hist.sum() + 1e-10)
        all_histograms[i] = aoi_hist

        H = _get_H(row, align_params)
        if H is None:
            pass1_records.append({
                "index": i, "filename": filename,
                "timestamp": timestamp, "status": "skip_no_homography",
                "threshold": None, "fit_prec": None,
                "fit_rec": None, "fit_f1": None,
                "source": None, "source_basename": None,
                "fit_pos": 0, "fit_neg": 0,
            })
            continue

        seg_mx, seg_my = _get_remap(row, seg_remap)
        warped = _load_warped_masks(basename, masks_dir, H, seg_mx, seg_my)
        valid_mask, sam_snow, sam_mountain = _build_region_masks(warped, mask)

        # Positive: SAM snow within valid region
        # Negative: SAM mountain (non-snow) within valid region
        pos_mask = sam_snow & valid_mask
        neg_mask = sam_mountain & ~sam_snow & valid_mask
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        if n_pos > 0 and n_neg > 0:
            thr, f_prec, f_rec, f_f1 = _best_threshold_from_hist(
                blue, pos_mask, neg_mask,
            )
            pass1_records.append({
                "index": i, "filename": filename,
                "timestamp": timestamp, "status": "learned",
                "threshold": int(thr), "fit_prec": f_prec,
                "fit_rec": f_rec, "fit_f1": f_f1,
                "source": "learned", "source_basename": basename,
                "fit_pos": n_pos, "fit_neg": n_neg,
            })
            learned_sources.append({
                "basename": basename,
                "timestamp": timestamp,
                "threshold": int(thr),
                "aoi_hist": aoi_hist,
            })
        else:
            pass1_records.append({
                "index": i, "filename": filename,
                "timestamp": timestamp, "status": "needs_fill",
                "threshold": None, "fit_prec": None,
                "fit_rec": None, "fit_f1": None,
                "source": None, "source_basename": None,
                "fit_pos": n_pos, "fit_neg": n_neg,
            })

    n_learned = sum(1 for r in pass1_records if r["status"] == "learned")
    n_needs_fill = sum(1 for r in pass1_records if r["status"] == "needs_fill")
    logger.info(
        f"Pass 1 complete: {n_learned} learned, "
        f"{n_needs_fill} need fill, "
        f"{n_images - n_learned - n_needs_fill} skipped"
    )

    # ---- Pass 2: Fill missing thresholds (histogram → nearest → fallback) ----
    for rec in pass1_records:
        if rec["status"] != "needs_fill":
            continue
        target_hist = all_histograms[rec["index"]]
        best_src = None
        source_type = "fallback"

        # Primary: histogram similarity matching
        if target_hist is not None and learned_sources:
            best_src = _best_histogram_source(target_hist, learned_sources)
            source_type = "histogram"

        # Fallback: temporal nearest
        if best_src is None and learned_sources:
            best_src = _nearest_threshold_source(
                rec["timestamp"], learned_sources,
            )
            source_type = "nearest"

        if best_src is not None:
            rec["threshold"] = int(best_src["threshold"])
            rec["source"] = source_type
            rec["source_basename"] = best_src["basename"]
            rec["status"] = f"filled_{source_type}"
        else:
            rec["threshold"] = fallback_threshold
            rec["source"] = "fallback"
            rec["source_basename"] = f"const_{fallback_threshold}"
            rec["status"] = "filled_fallback"

    # ---- Apply thresholds and produce snow masks ----
    logger.info("Applying BDN thresholds")
    snow_masks: dict[str, np.ndarray] = {}
    stats_records: list[dict] = []

    if output_dir is not None:
        out_root = Path(output_dir)
        preview_dir = out_root / "previews"
        mask_dir = out_root / "masks"
        preview_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

    for rec in tqdm(pass1_records, desc="snow:classify"):
        if rec["status"].startswith("skip"):
            continue

        filename = rec["filename"]
        basename = Path(filename).stem
        i = rec["index"]
        row = work_df.iloc[i]
        threshold = int(rec["threshold"])

        img = cv2.imread(str(aligned_dir / filename))
        if img is None:
            continue

        H = _get_H(row, align_params)
        if H is None:
            continue
        seg_mx, seg_my = _get_remap(row, seg_remap)
        warped = _load_warped_masks(basename, masks_dir, H, seg_mx, seg_my)
        valid_mask, sam_snow, sam_mountain = _build_region_masks(warped, mask)

        # BDN classification: Blue >= threshold within valid region
        blue = img[:, :, 0]
        snow_mask = (blue >= threshold) & valid_mask
        snow_masks[filename] = snow_mask

        n_valid = int(valid_mask.sum())
        n_snow = int(snow_mask.sum())
        n_sam_snow = int((sam_snow & valid_mask).sum())

        stats_records.append({
            "filename": filename,
            "timestamp": rec["timestamp"],
            "segment_id": row["segment_id"],
            "n_valid": n_valid,
            "n_snow": n_snow,
            "n_sam_snow": n_sam_snow,
            "snow_ratio": n_snow / n_valid if n_valid > 0 else 0.0,
            "threshold": threshold,
            "threshold_source": rec["source"],
            "source_basename": rec["source_basename"],
            "fit_status": rec["status"],
            "fit_prec": rec["fit_prec"],
            "fit_rec": rec["fit_rec"],
            "fit_f1": rec["fit_f1"],
        })

        if output_dir is not None:
            np.savez_compressed(
                str(mask_dir / f"{basename}_snow.npz"), mask=snow_mask,
            )
            _save_preview(
                img, snow_mask, valid_mask, sam_snow,
                str(preview_dir / f"{basename}.jpg"),
            )

    stats_df = pd.DataFrame(stats_records)
    if output_dir is not None and not stats_df.empty:
        stats_df.to_csv(str(Path(output_dir) / "snow_stats.csv"), index=False)

    # Log summary
    if len(stats_df) > 0:
        src_counts = stats_df["threshold_source"].value_counts().to_dict()
        mean_thr = stats_df["threshold"].mean()
        logger.info(
            f"Snow detection complete: {len(snow_masks)} images, "
            f"mean_threshold={mean_thr:.1f}, "
            f"source_distribution={src_counts}"
        )

    return snow_masks, stats_df


# =========================================================================
# BDN threshold helpers
# =========================================================================


def _best_threshold_from_hist(
    blue_u8: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
) -> tuple[int, float, float, float]:
    """Find the Blue threshold that maximises F1 between pos/neg masks.

    Sweeps thresholds in ``[_MIN_BDN, _MAX_BDN]`` using cumulative
    histograms for O(256) evaluation.

    Args:
        blue_u8: Blue channel image (uint8).
        pos_mask: Boolean mask of positive (snow) pixels.
        neg_mask: Boolean mask of negative (non-snow) pixels.

    Returns:
        ``(threshold, precision, recall, f1)`` at the best threshold.
    """
    pos_vals = blue_u8[pos_mask]
    neg_vals = blue_u8[neg_mask]
    pos_hist = np.bincount(pos_vals, minlength=256).astype(np.int64)
    neg_hist = np.bincount(neg_vals, minlength=256).astype(np.int64)

    # Cumulative from right: tp[t] = count of pos >= t
    tp_curve = np.cumsum(pos_hist[::-1])[::-1].astype(np.float64)
    fp_curve = np.cumsum(neg_hist[::-1])[::-1].astype(np.float64)
    fn_curve = float(pos_vals.size) - tp_curve

    t_slice = slice(_MIN_BDN, _MAX_BDN + 1)
    tp = tp_curve[t_slice]
    fp = fp_curve[t_slice]
    fn = fn_curve[t_slice]

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * prec * rec, prec + rec,
        out=np.zeros_like(prec), where=(prec + rec) > 0,
    )

    idx = int(np.argmax(f1))
    threshold = _MIN_BDN + idx
    return threshold, float(prec[idx]), float(rec[idx]), float(f1[idx])


def _best_histogram_source(
    target_hist: np.ndarray,
    learned: list[dict],
) -> dict | None:
    """Find the learned source with the most similar AOI blue histogram.

    Uses OpenCV histogram correlation (HISTCMP_CORREL) to compare
    normalised 256-bin histograms.

    Args:
        target_hist: Normalised histogram of the target image.
        learned: List of dicts with ``aoi_hist`` and ``threshold`` keys.

    Returns:
        The best matching dict, or ``None`` if no match found.
    """
    if not learned:
        return None
    best = None
    best_score = -2.0
    for src in learned:
        hist = src.get("aoi_hist")
        if hist is None:
            continue
        score = cv2.compareHist(
            target_hist.astype(np.float32),
            hist.astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        if score > best_score:
            best_score = score
            best = src
    return best


def _nearest_threshold_source(
    target_ts: pd.Timestamp | datetime,
    learned: list[dict],
) -> dict | None:
    """Return the learned source with the nearest timestamp to *target_ts*.

    Args:
        target_ts: Target timestamp to match against.
        learned: List of dicts with ``timestamp`` and ``threshold`` keys.

    Returns:
        The nearest dict, or ``None`` if *learned* is empty.
    """
    if not learned:
        return None
    return min(
        learned,
        key=lambda r: abs((r["timestamp"] - target_ts).total_seconds()),
    )


# =========================================================================
# Mask warping
# =========================================================================


def _warp_mask(
    mask: np.ndarray,
    H: np.ndarray,
    map_x: Optional[np.ndarray] = None,
    map_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Warp a boolean mask through distortion correction + homography.

    Args:
        mask: Boolean mask (H, W).
        H: 3×3 homography matrix.
        map_x: Distortion remap table (x), or None.
        map_y: Distortion remap table (y), or None.

    Returns:
        Warped boolean mask.
    """
    h, w = mask.shape[:2]
    mask_u8 = mask.astype(np.uint8) * 255
    if map_x is not None and map_y is not None:
        mask_u8 = cv2.remap(
            mask_u8, map_x, map_y, interpolation=cv2.INTER_NEAREST,
        )
    warped = cv2.warpPerspective(mask_u8, H, (w, h), flags=cv2.INTER_NEAREST)
    return warped > 127


def _get_H(row: pd.Series, align_params: dict) -> Optional[np.ndarray]:
    """Retrieve the homography matrix for a row's segment."""
    seg_id = str(int(row["segment_id"]))
    seg_p = align_params.get(seg_id, {})
    H_list = seg_p.get("H")
    if H_list is None:
        return None
    return np.array(H_list, dtype=np.float64)


def _load_warped_masks(
    basename: str,
    masks_dir: Path,
    H: np.ndarray,
    map_x: Optional[np.ndarray] = None,
    map_y: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Load SAM masks for *basename* and warp them with distortion + *H*.

    Returns:
        ``{prompt_name: warped_bool_mask}``.  Missing masks are omitted.
    """
    result: dict[str, np.ndarray] = {}
    for npz_path in masks_dir.glob(f"{basename}_*.npz"):
        prompt = npz_path.stem[len(basename) + 1:]   # e.g. "snow"
        mask = np.load(str(npz_path))["mask"]
        result[prompt] = _warp_mask(mask, H, map_x, map_y)
    return result


# =========================================================================
# Distortion remap tables
# =========================================================================


def _build_remap_tables(
    distortion_params: np.ndarray,
    img_height: int,
    img_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build distortion correction remap tables from parameters.

    Uses the same distortion model as ``align._distort_points``.

    Returns:
        ``(map_x, map_y)`` — float32 remap tables.
    """
    from yamami.align import _distort_points

    map_x, map_y = np.meshgrid(
        np.arange(img_width, dtype=np.float32),
        np.arange(img_height, dtype=np.float32),
    )
    grid = np.stack([map_x.flatten(), map_y.flatten()], axis=1)

    inv_params = -distortion_params
    grid_d = _distort_points(grid, inv_params, img_width, img_height)
    map_x_d = grid_d[:, 0].reshape(img_height, img_width).astype(np.float32)
    map_y_d = grid_d[:, 1].reshape(img_height, img_width).astype(np.float32)

    return map_x_d, map_y_d


def _build_segment_remap_tables(
    align_params: dict,
    img_height: int,
    img_width: int,
) -> dict[str, tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """Build per-segment distortion remap tables from alignment parameters.

    Returns:
        ``{segment_id_str: (map_x, map_y)}``.  Values are ``(None, None)``
        when distortion correction was not applied for that segment.
    """
    estimate_mode = align_params.get("estimate_distortion", "off")
    if estimate_mode == "off":
        return {}

    seg_maps: dict[str, tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}

    for key, seg_p in align_params.items():
        if not isinstance(seg_p, dict):
            continue
        dp = seg_p.get("distortion_params")
        if dp is not None:
            params = np.array(dp, dtype=np.float64)
            mx, my = _build_remap_tables(params, img_height, img_width)
            seg_maps[key] = (mx, my)

    return seg_maps


def _get_remap(
    row: pd.Series,
    seg_remap: dict[str, tuple[Optional[np.ndarray], Optional[np.ndarray]]],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Get distortion remap tables for a row's segment."""
    seg_id = str(int(row["segment_id"]))
    return seg_remap.get(seg_id, (None, None))


# =========================================================================
# Region masks
# =========================================================================


def _build_region_masks(
    warped_masks: dict[str, np.ndarray],
    aoi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the valid classification region, SAM snow, and SAM mountain masks.

    *valid_mask*    = AOI ∩ ¬(sky ∪ cloud ∪ fog ∪ …)
    *sam_snow*      = warped SAM snow mask
    *sam_mountain*  = warped SAM mountain mask

    Returns:
        ``(valid_mask, sam_snow, sam_mountain)`` — all boolean arrays.
    """
    exclude = np.zeros(aoi_mask.shape, dtype=bool)
    for prompt in _EXCLUDE_PROMPTS:
        if prompt in warped_masks:
            exclude |= warped_masks[prompt]

    valid_mask = aoi_mask.astype(bool) & ~exclude
    sam_snow = warped_masks.get(
        "snow", np.zeros(aoi_mask.shape, dtype=bool),
    )
    sam_mountain = warped_masks.get(
        "mountain", np.zeros(aoi_mask.shape, dtype=bool),
    )
    return valid_mask, sam_snow, sam_mountain


# =========================================================================
# Snow preview visualization
# =========================================================================


def _save_preview(
    img_bgr: np.ndarray,
    snow_mask: np.ndarray,
    valid_mask: np.ndarray,
    sam_snow: np.ndarray,
    output_path: str,
    preview_width: int = 2048,
) -> None:
    """Save side-by-side preview: original | snow overlay.

    Overlay colours:
        white  — classified as snow
        original (dim) — valid but not snow
        dark gray — excluded region (NA)
        cyan contour — SAM snow boundary
    """
    h, w = img_bgr.shape[:2]

    # Build overlay
    overlay = img_bgr.copy()

    # Dim excluded region
    excluded = ~valid_mask
    overlay[excluded] = (overlay[excluded] * 0.2).astype(np.uint8)

    # Dim non-snow valid region slightly
    non_snow_valid = valid_mask & ~snow_mask
    overlay[non_snow_valid] = (overlay[non_snow_valid] * 0.5).astype(np.uint8)

    # Snow pixels: bright white blend
    overlay[snow_mask, 0] = np.clip(
        overlay[snow_mask, 0].astype(np.int16) + 180, 0, 255,
    ).astype(np.uint8)
    overlay[snow_mask, 1] = np.clip(
        overlay[snow_mask, 1].astype(np.int16) + 180, 0, 255,
    ).astype(np.uint8)
    overlay[snow_mask, 2] = np.clip(
        overlay[snow_mask, 2].astype(np.int16) + 180, 0, 255,
    ).astype(np.uint8)

    # SAM snow boundary as cyan contour
    sam_u8 = (sam_snow & valid_mask).astype(np.uint8) * 255
    contours, _ = cv2.findContours(sam_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

    # Concatenate side by side
    combined = np.hstack([img_bgr, overlay])

    # Resize to preview width
    comb_h, comb_w = combined.shape[:2]
    if comb_w > preview_width:
        scale = preview_width / comb_w
        combined = cv2.resize(
            combined,
            (preview_width, int(comb_h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    cv2.imwrite(output_path, combined)


# =========================================================================
# High-reflectance rock detection
# =========================================================================


def detect_high_reflectance_rock(
    image: np.ndarray,
    aoi_mask: np.ndarray,
) -> np.ndarray:
    """Detect high-reflectance rocks to exclude from snow analysis.

    Args:
        image: RGB image as numpy array with shape (H, W, 3).
        aoi_mask: Binary mask defining the area of interest.

    Returns:
        Binary mask of rock pixels (1 = rock, 0 = not rock).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB with shape (H, W, 3)")

    if image.shape[:2] != aoi_mask.shape:
        raise ValueError("Image and AOI mask must have matching dimensions")

    brightness = image.sum(axis=2).astype(float)
    max_possible = 255 * 3
    high_threshold = 0.7 * max_possible

    high_reflectance = brightness >= high_threshold

    max_channel = image.max(axis=2).astype(float)
    min_channel = image.min(axis=2).astype(float)
    color_range = max_channel - min_channel
    low_saturation = color_range < 30

    rock_mask = (high_reflectance & low_saturation).astype(np.uint8)
    rock_mask = rock_mask * aoi_mask

    logger.debug(f"Detected {rock_mask.sum()} rock pixels")
    return rock_mask


# =========================================================================
# Snowmelt estimation — two-stage weighted change-point
# =========================================================================


def snowmelt(
    snow_masks: dict[str, np.ndarray],
    stats_df: pd.DataFrame,
    mask: np.ndarray,
    output_dir: Optional[str] = None,
    aligned_dir: Optional[str] = None,
    chunk_size: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-stage pixel-level snowmelt date estimation.

    Stage 1 computes per-image reliability weights from threshold source
    quality and isotonic regression of the image-level snow fraction
    (penalising images whose snow fraction exceeds the monotonic trend).

    Stage 2 aggregates observations to daily level and runs a weighted
    Bernoulli maximum-likelihood change-point detector per pixel.

    Args:
        snow_masks: ``{filename: bool_mask}`` from :func:`snow`.
        stats_df: DataFrame from :func:`snow` with columns
            ``filename``, ``timestamp``, ``snow_ratio``,
            ``threshold_source``.
        mask: Boolean AOI mask ``(H, W)``.
        output_dir: Save DOY map, confidence map, daily previews.
        aligned_dir: Directory with aligned images (for previews).
        chunk_size: AOI pixels per processing chunk.

    Returns:
        ``(doy_map, confidence_map)``

        *doy_map*: ``(H, W)`` float32.  Positive values are snowmelt
        day-of-year.  ``0`` = pixel never had snow.  ``-1`` = snow
        persisted through the observation period.  ``NaN`` = outside AOI.

        *confidence_map*: ``(H, W)`` float32 in ``[0, 1]``.
    """
    from tqdm import tqdm

    h, w = mask.shape[:2]
    aoi_rows, aoi_cols = np.where(mask > 0)
    n_aoi = len(aoi_rows)

    if n_aoi == 0 or len(snow_masks) == 0:
        logger.warning("No AOI pixels or snow masks — returning empty maps")
        return (
            np.full((h, w), np.nan, np.float32),
            np.full((h, w), np.nan, np.float32),
        )

    # Sort by timestamp, keep only images present in snow_masks
    work = stats_df[stats_df["filename"].isin(snow_masks)].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work = work.sort_values("timestamp").reset_index(drop=True)
    n_images = len(work)
    filenames = work["filename"].tolist()

    # Pre-sort masks into a list for fast iteration
    masks_list = [snow_masks[fn] for fn in filenames]

    # ---- Stage 1: Image-level reliability weights ----
    logger.info("Stage 1: Computing image reliability weights")
    weights = _compute_image_weights(work)

    # ---- Daily aggregation setup ----
    work["date"] = work["timestamp"].dt.date
    dates = sorted(work["date"].unique())
    n_days = len(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    doys = np.array([d.timetuple().tm_yday for d in dates])

    # Map each image to its day index
    img_day_idx = np.array([date_to_idx[d] for d in work["date"]])

    # Daily weights (sum of image weights per day)
    daily_w = np.zeros(n_days, dtype=np.float64)
    for i in range(n_images):
        daily_w[img_day_idx[i]] += weights[i]

    logger.info(
        f"Stage 2: Weighted change-point detection "
        f"({n_aoi} pixels, {n_days} days, {n_images} images)"
    )

    # ---- Stage 2: Per-pixel change-point in chunks ----
    doy_map = np.full((h, w), np.nan, dtype=np.float32)
    conf_map = np.zeros((h, w), dtype=np.float32)

    n_chunks = (n_aoi + chunk_size - 1) // chunk_size
    for ci in tqdm(range(n_chunks), desc="snowmelt:changepoint"):
        start = ci * chunk_size
        end = min(start + chunk_size, n_aoi)
        r_chunk = aoi_rows[start:end]
        c_chunk = aoi_cols[start:end]
        n_px = end - start

        # Extract chunk data from all images: (n_images, n_px)
        img_data = np.empty((n_images, n_px), dtype=np.float64)
        for i, m in enumerate(masks_list):
            img_data[i] = m[r_chunk, c_chunk]

        # Aggregate to daily level: daily_wy[d, px] = Σ w_i · y_i
        daily_wy = np.zeros((n_days, n_px), dtype=np.float64)
        for i in range(n_images):
            daily_wy[img_day_idx[i]] += weights[i] * img_data[i]

        # Run vectorised change-point
        best_k, p_high, p_low, confidence = _bernoulli_changepoint(
            daily_wy, daily_w,
        )

        # Convert to DOY and classify
        for j in range(n_px):
            ph = p_high[j]
            pl = p_low[j]
            k = best_k[j]
            if ph < 0.3:
                doy_map[r_chunk[j], c_chunk[j]] = 0        # no_snow
            elif pl > 0.7:
                doy_map[r_chunk[j], c_chunk[j]] = -1       # persistent
            else:
                doy_map[r_chunk[j], c_chunk[j]] = doys[k]
            conf_map[r_chunk[j], c_chunk[j]] = confidence[j]

    # ---- Output ----
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _save_doy_map(doy_map, conf_map, mask, str(out / "doy_map.png"))
        np.savez_compressed(
            str(out / "doy_map.npz"), doy=doy_map, confidence=conf_map,
        )
        if aligned_dir is not None:
            preview_dir = out / "daily_previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            _save_daily_previews(
                doy_map, mask, dates, doys,
                work, str(aligned_dir), str(preview_dir),
            )

    # Log summary
    valid = doy_map[mask > 0]
    n_melt = int(np.sum(valid > 0))
    n_no_snow = int(np.sum(valid == 0))
    n_persist = int(np.sum(valid == -1))
    logger.info(
        f"Snowmelt complete: melt={n_melt}, "
        f"no_snow={n_no_snow}, persistent={n_persist}"
    )
    if n_melt > 0:
        melt_doys = valid[valid > 0]
        logger.info(
            f"  DOY: median={np.median(melt_doys):.0f}, "
            f"min={np.min(melt_doys):.0f}, max={np.max(melt_doys):.0f}"
        )

    return doy_map, conf_map


# =========================================================================
# Snowmelt helpers
# =========================================================================


def _compute_image_weights(stats_df: pd.DataFrame) -> np.ndarray:
    """Per-image reliability weights for snowmelt estimation.

    Factors:
    1. *Threshold source*: ``learned`` > ``histogram`` > ``nearest``
       > ``fallback``.
    2. *Fit F1*: learned images with low F1 are down-weighted.
    3. *Isotonic residual*: images whose snow_ratio exceeds the
       monotonically non-increasing trend are penalised.
    """
    n = len(stats_df)
    w = np.ones(n, dtype=np.float64)

    # Factor 1: threshold source
    source_w = {
        "learned": 1.0, "histogram": 0.5,
        "nearest": 0.3, "fallback": 0.1,
    }
    sources = stats_df["threshold_source"].tolist()
    for i, src in enumerate(sources):
        w[i] *= source_w.get(src, 0.1)

    # Factor 2: fit_f1 (learned images only)
    if "fit_f1" in stats_df.columns:
        f1_vals = stats_df["fit_f1"].values
        for i in range(n):
            if sources[i] == "learned" and pd.notna(f1_vals[i]):
                w[i] *= max(0.5, float(f1_vals[i]))

    # Factor 3: isotonic residual
    snow_ratios = stats_df["snow_ratio"].values.astype(np.float64)
    iso_fit = _isotonic_non_increasing(snow_ratios)
    residual = np.maximum(0.0, snow_ratios - iso_fit)
    w *= np.exp(-5.0 * residual)

    # Normalise so mean ≈ 1
    w /= w.mean() + 1e-10
    return w


def _isotonic_non_increasing(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators for monotone non-increasing regression."""
    n = len(y)
    vals = y.astype(np.float64).copy()
    # blocks: list of [start, end, value, weight]
    blocks: list[list] = []
    for i in range(n):
        blocks.append([i, i + 1, vals[i], 1.0])
        while len(blocks) >= 2 and blocks[-2][2] < blocks[-1][2]:
            b1 = blocks[-2]
            b2 = blocks[-1]
            blocks.pop()
            blocks.pop()
            tw = b1[3] + b2[3]
            blocks.append([b1[0], b2[1],
                           (b1[2] * b1[3] + b2[2] * b2[3]) / tw, tw])
    out = np.empty(n, dtype=np.float64)
    for s, e, v, _ in blocks:
        out[s:e] = v
    return out


def _bernoulli_changepoint(
    daily_wy: np.ndarray,
    daily_w: np.ndarray,
    min_side_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised weighted Bernoulli ML change-point.

    For each pixel find the day index *k* that maximises the
    log-likelihood of a two-segment Bernoulli model (snow-probability
    *p_high* before *k*, *p_low* from *k* onward).

    Returns:
        ``(best_k, p_high, p_low, confidence)``
    """
    n_days, n_px = daily_wy.shape
    eps = 1e-10

    cum_wy = np.cumsum(daily_wy, axis=0)       # (n_days, n_px)
    cum_w = np.cumsum(daily_w)                   # (n_days,)
    total_wy = cum_wy[-1]                        # (n_px,)
    total_w = cum_w[-1]                           # scalar
    min_w = min_side_frac * total_w

    best_ll = np.full(n_px, -np.inf, dtype=np.float64)
    best_k = np.zeros(n_px, dtype=np.int32)
    best_ph = np.full(n_px, 0.5, dtype=np.float64)
    best_pl = np.full(n_px, 0.5, dtype=np.float64)

    for k in range(1, n_days):
        wb = cum_w[k - 1]
        wa = total_w - wb
        if wb < min_w or wa < min_w:
            continue

        wyb = cum_wy[k - 1]                     # (n_px,)
        wya = total_wy - wyb                     # (n_px,)

        ph = np.clip(wyb / (wb + eps), eps, 1 - eps)
        pl = np.clip(wya / (wa + eps), eps, 1 - eps)

        ll = (wyb * np.log(ph) + (wb - wyb) * np.log(1 - ph)
              + wya * np.log(pl) + (wa - wya) * np.log(1 - pl))

        better = ll > best_ll
        best_ll[better] = ll[better]
        best_k[better] = k
        best_ph[better] = ph[better]
        best_pl[better] = pl[better]

    # Confidence: LL improvement over single-Bernoulli null model
    p_null = np.clip(total_wy / (total_w + eps), eps, 1 - eps)
    ll_null = (total_wy * np.log(p_null)
               + (total_w - total_wy) * np.log(1 - p_null))
    delta_ll = np.maximum(0.0, best_ll - ll_null)
    conf = 1.0 - np.exp(-delta_ll / 5.0)
    conf *= np.clip((best_ph - best_pl) / 0.5, 0, 1)
    conf = np.clip(conf, 0, 1)

    return best_k, best_ph, best_pl, conf


# =========================================================================
# Snowmelt visualisation
# =========================================================================


def _save_doy_map(
    doy_map: np.ndarray,
    conf_map: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    preview_width: int = 2048,
) -> None:
    """Save DOY map as a colour-coded image with legend."""
    h, w = doy_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    valid_melt = (doy_map > 0) & ~np.isnan(doy_map) & (mask > 0)
    if valid_melt.any():
        melt_doys = doy_map[valid_melt]
        doy_min = float(np.nanmin(melt_doys))
        doy_max = float(np.nanmax(melt_doys))

        if doy_max > doy_min:
            norm = np.clip(
                (doy_map - doy_min) / (doy_max - doy_min) * 255,
                0, 255,
            ).astype(np.uint8)
        else:
            norm = np.full((h, w), 128, dtype=np.uint8)

        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        vis[valid_melt] = colored[valid_melt]

        # Legend text
        cv2.putText(
            vis,
            f"DOY {int(doy_min)} (early)",
            (10, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3,
        )
        cv2.putText(
            vis,
            f"DOY {int(doy_max)} (late)",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3,
        )

    # No snow: dark gray
    vis[(doy_map == 0) & (mask > 0)] = [50, 50, 50]
    # Persistent snow: white
    vis[(doy_map == -1) & (mask > 0)] = [255, 255, 255]

    # Resize
    if w > preview_width:
        scale = preview_width / w
        vis = cv2.resize(
            vis, (preview_width, int(h * scale)),
            interpolation=cv2.INTER_NEAREST,
        )

    cv2.imwrite(output_path, vis)
    logger.info(f"DOY map saved: {output_path}")


def _save_daily_previews(
    doy_map: np.ndarray,
    mask: np.ndarray,
    dates: list,
    doys: np.ndarray,
    work_df: pd.DataFrame,
    aligned_dir: str,
    output_dir: str,
    preview_width: int = 2048,
) -> None:
    """Save per-day preview: aligned image | model-predicted snow overlay."""
    from tqdm import tqdm

    aligned_path = Path(aligned_dir)
    out_path = Path(output_dir)
    h, w = doy_map.shape
    mask_bool = mask > 0

    # Pre-compute persistent snow mask
    persistent = (doy_map == -1) & mask_bool

    for i, date in tqdm(
        enumerate(dates), total=len(dates), desc="snowmelt:previews",
    ):
        doy = int(doys[i])

        # Pick one image for this day
        day_df = work_df[work_df["date"] == date]
        if day_df.empty:
            continue
        best_row = day_df.iloc[len(day_df) // 2]   # middle of day
        img = cv2.imread(str(aligned_path / best_row["filename"]))
        if img is None:
            continue

        # Model-predicted snow for this DOY
        model_snow = np.zeros((h, w), dtype=bool)
        melt_px = (doy_map > 0) & ~np.isnan(doy_map) & mask_bool
        model_snow[melt_px] = doy_map[melt_px] > doy
        model_snow[persistent] = True

        # Build overlay
        overlay = img.copy()
        non_aoi = ~mask_bool
        overlay[non_aoi] = (overlay[non_aoi] * 0.2).astype(np.uint8)

        # Snow: bright white-blue
        sp = model_snow & mask_bool
        overlay[sp, 0] = np.clip(
            overlay[sp, 0].astype(np.int16) + 150, 0, 255).astype(np.uint8)
        overlay[sp, 1] = np.clip(
            overlay[sp, 1].astype(np.int16) + 150, 0, 255).astype(np.uint8)
        overlay[sp, 2] = np.clip(
            overlay[sp, 2].astype(np.int16) + 150, 0, 255).astype(np.uint8)

        # Bare: dim
        bp = ~model_snow & mask_bool
        overlay[bp] = (overlay[bp] * 0.6).astype(np.uint8)

        combined = np.hstack([img, overlay])
        comb_h, comb_w = combined.shape[:2]
        if comb_w > preview_width:
            scale = preview_width / comb_w
            combined = cv2.resize(
                combined, (preview_width, int(comb_h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        cv2.putText(
            combined,
            f"DOY {doy}  {date}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        cv2.imwrite(
            str(out_path / f"day_{doy:03d}_{date}.jpg"),
            combined,
        )
