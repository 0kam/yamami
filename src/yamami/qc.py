"""
Quality control module for yamami.

Provides pre-alignment quality control functions to filter out unusable images
based on brightness and segmentation results (fog, sun, reflection),
plus ridgeline-based camera shift detection and segmentation.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from yamami.logging import get_logger

logger = get_logger(__name__)

# Reason code constants
REASON_TOO_DARK = "TOO_DARK"
REASON_TOO_BRIGHT = "TOO_BRIGHT"
REASON_HIGH_FOG = "HIGH_FOG"
REASON_HIGH_SUN = "HIGH_SUN"
REASON_HIGH_REFLECTION = "HIGH_REFLECTION"
REASON_LOW_MOUNTAIN = "LOW_MOUNTAIN"


def pre_qc(
    df: pd.DataFrame,
    luma_min: float = 20.0,
    luma_max: float = 240.0,
    fog_max: float = 0.01,
    sun_max: float = 0.01,
    reflection_max: float = 0.01,
    mountain_percentile: Optional[float] = None,
) -> pd.DataFrame:
    """
    Pre-alignment quality control.

    Filters images based on brightness and segmentation results before
    alignment. Images failing QC thresholds are marked as unusable with
    corresponding reason codes.

    Args:
        df: DataFrame with 'path', 'mean_luma', and optionally segment ratio
            columns (fog_ratio, sun_ratio, reflection_ratio, mountain_ratio).
        luma_min: Minimum acceptable mean luminance (0-255). Default is 20.
        luma_max: Maximum acceptable mean luminance (0-255). Default is 240.
        fog_max: Maximum acceptable fog ratio (0-1). Default is 0.01.
        sun_max: Maximum acceptable sun ratio (0-1). Default is 0.01.
        reflection_max: Maximum acceptable reflection ratio (0-1). Default is 0.01.
        mountain_percentile: Optional. If specified, images with mountain_ratio
            below this percentile of the dataset are flagged. E.g., 10 means
            images in the bottom 10% are flagged.

    Returns:
        pd.DataFrame: Input DataFrame with added columns:
            - is_usable (bool): Whether the image passes QC.
            - reason_codes (str): Comma-separated list of failure reasons.

    Example:
        >>> result = pre_qc(df)
        >>> result[result["is_usable"] == False]["reason_codes"]
        0    HIGH_FOG
        Name: reason_codes, dtype: object
    """
    merged = df.copy()

    # Initialize QC columns
    is_usable = pd.Series([True] * len(merged), index=merged.index)
    reason_codes = pd.Series([""] * len(merged), index=merged.index)

    # Check brightness thresholds using mean_luma (0-255)
    if "mean_luma" in merged.columns:
        too_dark = merged["mean_luma"] < luma_min
        too_bright = merged["mean_luma"] > luma_max

        is_usable = is_usable & ~too_dark & ~too_bright
        reason_codes = _append_reason(reason_codes, too_dark, REASON_TOO_DARK)
        reason_codes = _append_reason(reason_codes, too_bright, REASON_TOO_BRIGHT)

    # Check fog ratio
    if "fog_ratio" in merged.columns:
        high_fog = merged["fog_ratio"] > fog_max
        is_usable = is_usable & ~high_fog
        reason_codes = _append_reason(reason_codes, high_fog, REASON_HIGH_FOG)

    # Check sun ratio
    if "sun_ratio" in merged.columns:
        high_sun = merged["sun_ratio"] > sun_max
        is_usable = is_usable & ~high_sun
        reason_codes = _append_reason(reason_codes, high_sun, REASON_HIGH_SUN)

    # Check reflection ratio
    if "reflection_ratio" in merged.columns:
        high_reflection = merged["reflection_ratio"] > reflection_max
        is_usable = is_usable & ~high_reflection
        reason_codes = _append_reason(reason_codes, high_reflection, REASON_HIGH_REFLECTION)

    # Optional: Check mountain percentile
    if mountain_percentile is not None and "mountain_ratio" in merged.columns:
        threshold = np.percentile(merged["mountain_ratio"].dropna(), mountain_percentile)
        low_mountain = merged["mountain_ratio"] < threshold
        is_usable = is_usable & ~low_mountain
        reason_codes = _append_reason(reason_codes, low_mountain, REASON_LOW_MOUNTAIN)
        logger.info(f"Mountain percentile threshold ({mountain_percentile}%): {threshold:.4f}")

    # Build result DataFrame
    result = merged.copy()
    result["is_usable"] = is_usable.values
    result["reason_codes"] = reason_codes.values

    usable_count = result["is_usable"].sum()
    logger.info(
        f"Pre-QC: {usable_count}/{len(result)} images passed "
        f"(luma=[{luma_min:.0f},{luma_max:.0f}], "
        f"fog<{fog_max:.2f}, sun<{sun_max:.2f}, reflection<{reflection_max:.2f})"
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


def _put_text(img, text, org, fg=(0, 0, 0), scale=0.5, thickness=1):
    """Draw outlined text (white outline + foreground color)."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness + 1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                fg, thickness)


# =============================================================================
# Ridge QC
# =============================================================================


def ridge_qc(
    df: pd.DataFrame,
    masks_dir,
    target_image,
    output_dir=None,
    # Basic QC params
    luma_min: float = 20.0,
    luma_max: float = 240.0,
    fog_max: float = 0.05,
    sun_max: float = 0.01,
    reflection_max: float = 0.01,
    # Ridgeline params
    search_margin: int = 10,
    blue_ratio_threshold: float = 0.34,
    fog_skip_threshold: float = 0.05,
    # Matching params
    inlier_threshold: float = 3.0,
    inlier_ratio_threshold: float = 0.8,
    # Segmentation params
    displacement_threshold: float = 5.0,
) -> pd.DataFrame:
    """Ridgeline-based quality control, shift detection, and segmentation.

    Combines basic threshold QC (via :func:`pre_qc`), adaptive ridgeline
    detection, RANSAC matching against a target image, fog-skip logic,
    and camera-shift segment detection.

    Args:
        df: DataFrame with ``path``, ``filename``, ``mean_luma``, and
            segment ratio columns from previous pipeline steps.
        masks_dir: Directory containing SAM mask .npz files.
        target_image: Path to the reference image for ridgeline matching.
        output_dir: Optional output directory for grid visualization.
        luma_min: Minimum acceptable mean luminance.
        luma_max: Maximum acceptable mean luminance.
        fog_max: Maximum fog ratio for basic QC.
        sun_max: Maximum sun ratio for basic QC.
        reflection_max: Maximum reflection ratio for basic QC.
        search_margin: Pixel margin around SAM ridge for detection.
        blue_ratio_threshold: Blue ratio threshold for method selection.
        fog_skip_threshold: Fog ratio threshold to skip ridgeline matching.
        inlier_threshold: RANSAC inlier residual threshold (pixels).
        inlier_ratio_threshold: Min inlier ratio for match success.
        displacement_threshold: Corner displacement threshold for segments.

    Returns:
        DataFrame with added columns: ``is_usable``, ``reason_codes``,
        ``ridge_method``, ``ridge_blue_ratio``, ``ridge_dx``, ``ridge_dy``,
        ``ridge_angle``, ``ridge_scale``, ``ridge_error_median``,
        ``ridge_match_success``, ``segment_id``, ``ridge_confirmed``,
        ``ridge_prev_displacement``.
    """
    from yamami.ridgeline import (
        detect_ridgeline,
        detect_segments,
        get_sam_ridge,
        load_mask,
        match_ridgelines_ransac,
    )
    from yamami.viz import create_grid_image, draw_ridgeline

    masks_dir = Path(masks_dir)
    target_path = Path(target_image)

    # --- Step 1: Basic QC ---
    result_df = pre_qc(
        df,
        luma_min=luma_min,
        luma_max=luma_max,
        fog_max=fog_max,
        sun_max=sun_max,
        reflection_max=reflection_max,
    )

    # --- Step 2: Target ridgeline ---
    target_img = cv2.imread(str(target_path))
    if target_img is None:
        raise FileNotFoundError(f"Target image not found: {target_path}")

    target_basename = target_path.stem
    target_sam_ridge = get_sam_ridge(masks_dir / f"{target_basename}_mountain.npz")
    target_sky_mask = load_mask(masks_dir / f"{target_basename}_sky.npz")
    target_cloud_mask = load_mask(masks_dir / f"{target_basename}_cloud.npz")

    target_ridge, _target_method, _target_br = detect_ridgeline(
        target_img, target_sam_ridge, target_sky_mask, target_cloud_mask,
        search_margin=search_margin,
        blue_ratio_threshold=blue_ratio_threshold,
    )
    img_h, img_w = target_img.shape[:2]
    h_approx = (
        int(np.max(target_ridge[target_ridge >= 0]))
        if np.any(target_ridge >= 0)
        else img_h
    )
    ridge_cy = h_approx / 2

    # --- Step 3: Per-image ridgeline detection + matching ---
    match_results = []  # parallel list for detect_segments
    grid_thumbs = []  # (thumb, index) pairs for post-annotation

    thumb_scale = 0.15

    for _, row in result_df.iterrows():
        filepath = row["path"]
        basename = Path(filepath).stem
        is_target = (
            filepath == str(target_path)
            or Path(filepath).name == target_path.name
        )

        default = {
            "filename": row["filename"],
            "method": "skip",
            "blue_ratio": np.nan,
            "dx": np.nan, "dy": np.nan,
            "angle": np.nan, "scale": np.nan,
            "error_median": np.nan,
            "match_success": False,
            "prev_displacement": np.nan,
        }

        img = cv2.imread(filepath)
        if img is None:
            match_results.append(default)
            continue

        # Fog skip
        fog_ratio = row.get("fog_ratio", 0.0)
        if isinstance(fog_ratio, float) and fog_ratio >= fog_skip_threshold:
            default["method"] = "fog_skip"
            match_results.append(default)
            if output_dir is not None:
                thumb = cv2.resize(img, None, fx=thumb_scale, fy=thumb_scale)
                ts = str(row.get("timestamp", ""))
                label = f"{ts[:10]} {ts[11:16]}" if len(ts) >= 16 else basename
                _put_text(thumb, label, (10, 25))
                _put_text(thumb, f"fog:{fog_ratio:.0%}", (10, 45),
                          fg=(0, 128, 255))
                grid_thumbs.append((thumb, len(match_results) - 1))
            continue

        sam_ridge = get_sam_ridge(masks_dir / f"{basename}_mountain.npz")
        sky_mask = load_mask(masks_dir / f"{basename}_sky.npz")
        cloud_mask = load_mask(masks_dir / f"{basename}_cloud.npz")

        ridge, method_name, blue_ratio = detect_ridgeline(
            img, sam_ridge, sky_mask, cloud_mask,
            search_margin=search_margin,
            blue_ratio_threshold=blue_ratio_threshold,
        )

        mr = match_ridgelines_ransac(
            target_ridge, ridge,
            inlier_threshold=inlier_threshold,
            inlier_ratio_threshold=inlier_ratio_threshold,
        )

        match_results.append({
            "filename": row["filename"],
            "method": method_name,
            "blue_ratio": blue_ratio,
            "dx": mr["dx"],
            "dy": mr["dy"],
            "angle": mr["angle"],
            "scale": mr["scale"],
            "error_median": mr["error_median"],
            "match_success": mr["match_success"],
            "prev_displacement": np.nan,
        })

        # Grid visualization
        if output_dir is not None:
            vis = draw_ridgeline(
                img, ridge, thickness=8,
                residual_mask=mr["residual_mask"],
                residual_threshold=inlier_threshold,
            )
            thumb = cv2.resize(vis, None, fx=thumb_scale, fy=thumb_scale)

            # Labels
            ts = str(row.get("timestamp", ""))
            label = f"{ts[:10]} {ts[11:16]}" if len(ts) >= 16 else basename
            _put_text(thumb, label, (10, 25))

            em = mr["error_median"]
            em_str = f"err:{em:.1f}" if not np.isinf(em) else "err:inf"
            _put_text(thumb, em_str, (10, 45))

            if is_target:
                _put_text(thumb, "TARGET", (10, 65), fg=(0, 255, 255))
                cv2.rectangle(
                    thumb, (0, 0),
                    (thumb.shape[1] - 1, thumb.shape[0] - 1),
                    (0, 255, 255), 4,
                )
            else:
                status = f"dx:{mr['dx']:.0f} dy:{mr['dy']:.0f}"
                color = (
                    (0, 255, 0) if mr["match_success"] else (0, 0, 255)
                )
                _put_text(thumb, status, (10, 65), fg=color)

            grid_thumbs.append((thumb, len(match_results) - 1))

    # --- Step 4: Segment detection ---
    segment_info = detect_segments(
        match_results, img_w, img_h, ridge_cy,
        displacement_threshold=displacement_threshold,
    )

    # --- Step 5: Annotate segment IDs on grid thumbnails ---
    if output_dir is not None:
        for thumb, idx in grid_thumbs:
            seg_id = match_results[idx].get("segment_id", "")
            if seg_id != "":
                _put_text(thumb, f"seg:{seg_id}", (10, 85), fg=(255, 200, 0))

    # --- Step 6: Write columns back to DataFrame ---
    result_df["ridge_method"] = [r["method"] for r in match_results]
    result_df["ridge_blue_ratio"] = [r["blue_ratio"] for r in match_results]
    result_df["ridge_dx"] = [r["dx"] for r in match_results]
    result_df["ridge_dy"] = [r["dy"] for r in match_results]
    result_df["ridge_angle"] = [r["angle"] for r in match_results]
    result_df["ridge_scale"] = [r["scale"] for r in match_results]
    result_df["ridge_error_median"] = [r["error_median"] for r in match_results]
    result_df["ridge_match_success"] = [r["match_success"] for r in match_results]
    result_df["segment_id"] = [r["segment_id"] for r in match_results]
    result_df["ridge_confirmed"] = [r["ridge_confirmed"] for r in match_results]
    result_df["ridge_prev_displacement"] = [
        r.get("prev_displacement", np.nan) for r in match_results
    ]

    logger.info(
        f"Ridge QC: {sum(r['match_success'] for r in match_results)}/{len(match_results)} "
        f"matched, {len(segment_info)} segments"
    )

    # --- Step 7: Optional grid output (7x7 pages) ---
    if output_dir is not None and grid_thumbs:
        out = Path(output_dir)
        ridgeline_dir = out / "ridgeline"
        ridgeline_dir.mkdir(parents=True, exist_ok=True)

        images = [t for t, _ in grid_thumbs]
        grid_cols = 7
        per_page = grid_cols * grid_cols  # 49

        for page_idx in range(0, len(images), per_page):
            page_images = images[page_idx : page_idx + per_page]
            grid = create_grid_image(page_images, cols=grid_cols)

            # Title bar
            page_num = page_idx // per_page + 1
            total_pages = (len(images) + per_page - 1) // per_page
            title = (
                f"[{page_num}/{total_pages}] "
                "Green=inlier, Red=outlier, Gray=no match, Yellow=TARGET"
            )
            title_bar = np.zeros((50, grid.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_bar, title, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            grid = np.vstack([title_bar, grid])

            grid_path = ridgeline_dir / f"ridgeline_grid_{page_num:03d}.jpg"
            cv2.imwrite(str(grid_path), grid)

        logger.info(
            f"Ridge grid saved: {total_pages} page(s) in {ridgeline_dir}"
        )

    return result_df
