"""
Image alignment module for yamami.

Provides round-based image alignment with lens distortion correction
and ridgeline-based validation. Leverages segment/shift information
from ridge_qc (Step 3) to select anchor candidates per segment.
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize

from yamami.logging import get_logger

logger = get_logger(__name__)

# List of methods that require the imm package
IMM_METHODS = [
    "sift-lightglue",
    "superpoint-lightglue",
    "minima-superpoint-lightglue",
    "roma",
    "tiny-roma",
    "minima-roma",
    "loftr",
    "minima-loftr",
    "ufm",
    "rdd",
    "master",
]

# Lightweight methods (LightGlue-based) that can handle full-resolution images
_LIGHTGLUE_METHODS = [
    "sift-lightglue",
    "superpoint-lightglue",
    "minima-superpoint-lightglue",
]

# Default resize for memory-intensive methods
_DEFAULT_RESIZE_HEAVY = 640

# Default Lowe's ratio test threshold
_DEFAULT_LOWE_RATIO = 0.7


# =============================================================================
# Public API
# =============================================================================


def align(
    df: pd.DataFrame,
    target_image: str,
    masks_dir: str,
    method: str = "superpoint-lightglue",
    output_dir: Optional[str] = None,
    device: str = "cpu",
    resize: Optional[int] = None,
    max_rounds: int = 5,
    min_ransac_inliers: int = 20,
    max_rmse: float = 5.0,
    ridge_validation: bool = True,
    estimate_distortion: str = "global",
    grid_size: int = 100,
    ransac_thresh: float = 10.0,
) -> pd.DataFrame:
    """Round-based image alignment with lens distortion correction.

    Uses segment and ridge information from ridge_qc (Step 3) to select
    anchor candidates, estimate lens distortion, and iteratively align
    all segments to the target image coordinate system.

    Args:
        df: DataFrame from ridge_qc with ``segment_id``,
            ``ridge_match_success``, ``ridge_blue_ratio``, etc.
        target_image: Path to the target (reference) image.
        masks_dir: Directory containing SAM mask .npz files.
        method: Feature matching method (default ``"superpoint-lightglue"``).
        output_dir: Base output directory. Aligned images are saved under
            ``<output_dir>/aligned/``.
        device: Device for IMM methods (``"cpu"`` or ``"cuda"``).
        resize: Resize for IMM methods (None = auto).
        max_rounds: Maximum alignment rounds.
        min_ransac_inliers: Minimum RANSAC inlier count (Layer 1).
        max_rmse: Maximum reprojection RMSE in pixels (Layer 2).
        ridge_validation: Whether to use ridgeline validation (Layer 3).
        estimate_distortion: Lens distortion estimation mode.
            ``"global"`` estimates one set of parameters from all anchors
            (assumes same camera).  ``"per_segment"`` estimates distortion
            independently for each segment.  ``"off"`` disables distortion
            estimation entirely.
        grid_size: Grid cell size in pixels for spatial thinning of matches.
            Set to 0 to disable.
        ransac_thresh: RANSAC reprojection threshold in pixels for
            homography estimation.

    Returns:
        DataFrame with added columns: ``align_status``, ``align_rmse``,
        ``align_num_matches``, ``align_anchor``, ``align_round``,
        ``align_method``.
    """
    from yamami.ridgeline import (
        compute_blue_ratio,
        detect_ridgeline,
        get_sam_ridge,
        load_mask,
        match_ridgelines_ransac,
    )

    target_path = Path(target_image)
    masks_dir = Path(masks_dir)

    if output_dir is None:
        output_dir = "."
    aligned_dir = Path(output_dir) / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    result_df = df.copy()

    # --- Detect target ridgeline for Layer 3 validation ---
    target_img = cv2.imread(str(target_path))
    if target_img is None:
        raise FileNotFoundError(f"Target image not found: {target_path}")

    img_h, img_w = target_img.shape[:2]
    target_ridge = None

    if ridge_validation:
        target_basename = target_path.stem
        target_sam_ridge = get_sam_ridge(
            masks_dir / f"{target_basename}_mountain.npz"
        )
        target_sky_mask = load_mask(masks_dir / f"{target_basename}_sky.npz")
        target_cloud_mask = load_mask(
            masks_dir / f"{target_basename}_cloud.npz"
        )
        target_ridge, _, target_blue_ratio = detect_ridgeline(
            target_img,
            target_sam_ridge,
            target_sky_mask,
            target_cloud_mask,
        )
    else:
        target_blue_ratio = 0.33

    # --- Select anchor candidates per segment ---
    segment_ids = sorted(result_df["segment_id"].dropna().unique())
    segment_candidates = {}

    for seg_id in segment_ids:
        candidates = _select_anchor_candidates(
            result_df, int(seg_id), target_blue_ratio,
        )
        segment_candidates[int(seg_id)] = candidates
        logger.info(
            f"Segment {int(seg_id)}: {len(candidates)} anchor candidates"
        )

    # --- Detect target segment (skip alignment for it) ---
    target_segment_id: Optional[int] = None
    target_name = target_path.name
    for seg_id in segment_ids:
        seg_df = result_df[result_df["segment_id"] == seg_id]
        seg_names = seg_df["path"].apply(lambda p: Path(p).name)
        if target_name in seg_names.values:
            target_segment_id = int(seg_id)
            segment_candidates.pop(target_segment_id, None)
            logger.info(
                f"Segment {target_segment_id}: contains target image, "
                "skipping alignment (identity)"
            )
            break

    # --- Estimate lens distortion ---
    # seg_maps: {segment_id: (map_x, map_y)} — per-segment distortion maps
    # distortion_params_dict: {segment_id: params} — for saving to JSON
    seg_maps: dict[int, tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
    distortion_params_dict: dict[int, Optional[np.ndarray]] = {}

    if estimate_distortion == "global":
        # Estimate once from all anchors
        all_anchor_paths = []
        for cands in segment_candidates.values():
            all_anchor_paths.extend([c["path"] for c in cands])

        if len(all_anchor_paths) > 0:
            params, mx, my = _estimate_lens_distortion(
                all_anchor_paths,
                str(target_path),
                method=method,
                device=device,
                resize=resize,
                img_shape=(img_h, img_w),
            )
            for seg_id in segment_candidates:
                seg_maps[seg_id] = (mx, my)
                distortion_params_dict[seg_id] = params

    elif estimate_distortion == "per_segment":
        # Estimate independently per segment
        for seg_id, cands in segment_candidates.items():
            anchor_paths = [c["path"] for c in cands]
            if len(anchor_paths) == 0:
                seg_maps[seg_id] = (None, None)
                distortion_params_dict[seg_id] = None
                continue

            params, mx, my = _estimate_lens_distortion(
                anchor_paths,
                str(target_path),
                method=method,
                device=device,
                resize=resize,
                img_shape=(img_h, img_w),
            )
            seg_maps[seg_id] = (mx, my)
            distortion_params_dict[seg_id] = params

    elif estimate_distortion != "off":
        raise ValueError(
            f"Unknown estimate_distortion '{estimate_distortion}'. "
            "Use 'global', 'per_segment', or 'off'."
        )

    # --- Round-based alignment ---
    segment_results = _round_based_alignment(
        result_df,
        str(target_path),
        segment_candidates,
        method=method,
        device=device,
        resize=resize,
        max_rounds=max_rounds,
        min_ransac_inliers=min_ransac_inliers,
        max_rmse=max_rmse,
        ridge_validation=ridge_validation,
        target_ridge=target_ridge,
        masks_dir=masks_dir,
        aligned_dir=str(aligned_dir),
        seg_maps=seg_maps,
        grid_size=grid_size,
        ransac_thresh=ransac_thresh,
    )

    # --- Register target segment as identity (no warp needed) ---
    if target_segment_id is not None:
        H_identity = np.eye(3, dtype=np.float64)
        segment_results[target_segment_id] = {
            "status": "success",
            "H": H_identity,
            "rmse": 0.0,
            "num_matches": 0,
            "anchor": target_name,
            "round": 0,
            "method": "identity",
        }
        # Copy images as-is (no distortion correction, no warp)
        _warp_segment_images(
            result_df,
            target_segment_id,
            H_identity,
            None,
            None,
            str(aligned_dir),
        )

    # --- Apply warps to all images in successful segments ---
    for seg_id, seg_result in segment_results.items():
        if seg_id == target_segment_id:
            continue  # Already handled above
        if seg_result["status"] == "success":
            mx, my = seg_maps.get(seg_id, (None, None))
            _warp_segment_images(
                result_df,
                seg_id,
                seg_result["H"],
                mx,
                my,
                str(aligned_dir),
            )

    # --- Save match visualizations ---
    from yamami.viz import plot_match_result

    match_viz_dir = aligned_dir / "match_viz"
    for seg_id, sr in segment_results.items():
        pts_src_inlier = sr.get("pts_src_inlier")
        pts_dst_inlier = sr.get("pts_dst_inlier")
        if pts_src_inlier is None or pts_dst_inlier is None:
            continue
        if sr.get("H") is None:
            continue

        # Find anchor path from segment candidates
        anchor_path = None
        for c in segment_candidates.get(seg_id, []):
            if Path(c["path"]).name == sr["anchor"]:
                anchor_path = c["path"]
                break
        if anchor_path is None:
            continue

        ref_path_used = sr.get("ref_path", str(target_path))
        status_tag = "FAILED" if sr["status"] != "success" else ""
        fail_reason = sr.get("fail_reason", "")
        viz_path = str(match_viz_dir / f"seg{seg_id}_match.jpg")

        plot_match_result(
            anchor_path=anchor_path,
            ref_path=ref_path_used,
            pts_src=pts_src_inlier,
            pts_dst=pts_dst_inlier,
            H=sr["H"],
            segment_id=seg_id,
            round_num=sr.get("round", -1),
            rmse=sr.get("rmse", 0.0),
            num_matches=sr.get("num_matches", 0),
            method=sr.get("method", method),
            output_path=viz_path,
            status_tag=status_tag,
            fail_reason=fail_reason,
        )
        logger.info(f"Saved match visualization: {viz_path}")

    # --- Write output columns ---
    align_status = []
    align_rmse = []
    align_num_matches = []
    align_anchor = []
    align_round = []
    align_method = []

    for _, row in result_df.iterrows():
        seg_id = row.get("segment_id")
        if pd.notna(seg_id) and int(seg_id) in segment_results:
            sr = segment_results[int(seg_id)]
            align_status.append(sr["status"])
            align_rmse.append(sr.get("rmse", np.nan))
            align_num_matches.append(sr.get("num_matches", 0))
            align_anchor.append(sr.get("anchor", ""))
            align_round.append(sr.get("round", -1))
            align_method.append(sr.get("method", method))
        else:
            align_status.append("failed")
            align_rmse.append(np.nan)
            align_num_matches.append(0)
            align_anchor.append("")
            align_round.append(-1)
            align_method.append(method)

    result_df["align_status"] = align_status
    result_df["align_rmse"] = align_rmse
    result_df["align_num_matches"] = align_num_matches
    result_df["align_anchor"] = align_anchor
    result_df["align_round"] = align_round
    result_df["align_method"] = align_method

    # --- Save alignment parameters ---
    params_data = {}
    for seg_id, sr in segment_results.items():
        params_data[str(seg_id)] = {
            "status": sr["status"],
            "H": sr["H"].tolist() if sr.get("H") is not None else None,
            "rmse": sr.get("rmse"),
            "num_matches": sr.get("num_matches"),
            "anchor": sr.get("anchor"),
            "round": sr.get("round"),
        }
    for seg_id, dp in distortion_params_dict.items():
        if dp is not None:
            seg_key = str(seg_id)
            if seg_key in params_data:
                params_data[seg_key]["distortion_params"] = dp.tolist()
            else:
                params_data[f"distortion_{seg_key}"] = dp.tolist()
    params_data["estimate_distortion"] = estimate_distortion

    params_path = aligned_dir / "alignment_params.json"
    with open(params_path, "w") as f:
        json.dump(params_data, f, indent=2, default=str)

    n_success = sum(
        1 for s in segment_results.values() if s["status"] == "success"
    )
    logger.info(
        f"Alignment complete: {n_success}/{len(segment_results)} segments "
        f"aligned, saved to {aligned_dir}"
    )

    return result_df


# =============================================================================
# Anchor candidate selection
# =============================================================================


def _select_anchor_candidates(
    df: pd.DataFrame,
    segment_id: int,
    target_blue_ratio: float,
    top_n: int = 3,
) -> list[dict]:
    """Select best anchor candidates for a segment.

    Scores images with ``ridge_match_success == True`` using:
      - blue_ratio proximity to target (weight 0.5)
      - timestamp proximity to noon (weight 0.2)
      - middle of segment preference (weight 0.3)

    Args:
        df: DataFrame with ridge_qc columns.
        segment_id: Segment to select from.
        target_blue_ratio: Blue ratio of the target image.
        top_n: Number of candidates to return.

    Returns:
        List of dicts with ``path``, ``score``, ``blue_ratio``.
    """
    seg_df = df[df["segment_id"] == segment_id].copy()
    matched = seg_df[seg_df["ridge_match_success"] == True]  # noqa: E712

    if len(matched) == 0:
        # Fallback: use all usable images in segment
        if "is_usable" in seg_df.columns:
            matched = seg_df[seg_df["is_usable"] == True]  # noqa: E712
        if len(matched) == 0:
            matched = seg_df

    scores = []
    for idx, row in matched.iterrows():
        # Blue ratio proximity score (0-1, higher is better)
        br = row.get("ridge_blue_ratio", 0.33)
        if pd.isna(br):
            br = 0.33
        br_score = 1.0 - min(abs(br - target_blue_ratio) / 0.2, 1.0)

        # Noon proximity score (0-1)
        noon_score = 0.5  # default
        ts = row.get("timestamp")
        if pd.notna(ts):
            try:
                ts_parsed = pd.Timestamp(ts)
                hour = ts_parsed.hour + ts_parsed.minute / 60.0
                noon_score = 1.0 - min(abs(hour - 12.0) / 6.0, 1.0)
            except Exception:
                pass

        # Position in segment score (prefer middle)
        seg_indices = matched.index.tolist()
        if len(seg_indices) > 1:
            pos = seg_indices.index(idx) / (len(seg_indices) - 1)
            mid_score = 1.0 - abs(pos - 0.5) * 2.0
        else:
            mid_score = 1.0

        score = 0.5 * br_score + 0.2 * noon_score + 0.3 * mid_score
        scores.append(
            {
                "path": row["path"],
                "score": score,
                "blue_ratio": br,
                "idx": idx,
            }
        )

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_n]


# =============================================================================
# Lens distortion estimation
# =============================================================================


def _distort_points(
    pts: np.ndarray,
    params: np.ndarray,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """Apply lens distortion model to points.

    Brown-Conrady distortion model with 8 parameters:
    k1-k3 (radial numerator), p1-p2 (tangential), k4-k6 (radial denominator).

    Args:
        pts: Points array of shape (N, 2).
        params: 8-element distortion parameter array.
        img_width: Image width.
        img_height: Image height.

    Returns:
        Distorted points array of shape (N, 2).
    """
    k1, k2, k3, p1, p2, k4, k5, k6 = params

    centre = np.array(
        [(img_width - 1) / 2.0, (img_height - 1) / 2.0], dtype=np.float64
    )
    x1 = (pts[:, 0] - centre[0]) / centre[0]
    y1 = (img_height / img_width) * (pts[:, 1] - centre[1]) / centre[1]

    r2 = x1**2 + y1**2
    r4 = r2**2
    r6 = r2 * r4

    radial_num = 1 + k1 * r2 + k2 * r4 + k3 * r6
    radial_den = 1 + k4 * r2 + k5 * r4 + k6 * r6

    x1_d = x1 * radial_num / radial_den + 2 * p1 * x1 * y1 + p2 * (r2 + 2 * x1**2)
    y1_d = y1 * radial_num / radial_den + 2 * p2 * x1 * y1 + p1 * (r2 + 2 * y1**2)

    x1_d = x1_d * centre[0] + centre[0]
    y1_d = (img_width / img_height) * y1_d * centre[1] + centre[1]

    return np.stack([x1_d, y1_d], axis=1)


def _estimate_lens_distortion(
    anchor_paths: list[str],
    target_path: str,
    method: str,
    device: str,
    resize: Optional[int],
    img_shape: tuple[int, int],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate lens distortion from anchor-target match points.

    Collects feature matches from multiple anchor images against the
    target, then optimizes Brown-Conrady distortion parameters to
    minimise the mean per-anchor reprojection error.  Each anchor gets
    its own homography so that different camera positions (segments)
    do not interfere with each other.

    Args:
        anchor_paths: List of anchor image paths.
        target_path: Target image path.
        method: Matching method.
        device: Compute device.
        resize: Resize parameter.
        img_shape: (height, width) of images.

    Returns:
        Tuple of (distortion_params, map_x, map_y). All None if
        estimation is skipped or fails.
    """
    img_h, img_w = img_shape

    # Collect match points per anchor (keep separate)
    anchor_point_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for anchor_path in anchor_paths:
        try:
            pts_src, pts_dst = _match_images(
                anchor_path, target_path, method=method, device=device,
                resize=resize,
            )
            if len(pts_src) >= 10:
                anchor_point_pairs.append(
                    (pts_src.astype(np.float64), pts_dst.astype(np.float64))
                )
        except Exception as e:
            logger.debug(f"Distortion estimation: skipping {anchor_path}: {e}")

    if len(anchor_point_pairs) == 0:
        logger.info("Distortion estimation: no match points, skipping")
        return None, None, None

    total_pts = sum(len(p[0]) for p in anchor_point_pairs)
    logger.info(
        f"Distortion estimation: {total_pts} total match points "
        f"from {len(anchor_point_pairs)} anchors"
    )

    if total_pts < 30:
        logger.info("Distortion estimation: insufficient points, skipping")
        return None, None, None

    # Baseline: mean per-anchor RMSE (homography only, no distortion)
    base_rmses = []
    for pts_src, pts_dst in anchor_point_pairs:
        H_base, base_mask = estimate_homography(
            pts_src.astype(np.float32), pts_dst.astype(np.float32),
        )
        if H_base is None:
            continue
        rmse, _ = _compute_rmse(H_base, pts_src, pts_dst, inlier_mask=base_mask)
        base_rmses.append(rmse)

    if len(base_rmses) == 0:
        return None, None, None

    base_rmse = float(np.mean(base_rmses))
    logger.info(f"Distortion estimation: baseline RMSE = {base_rmse:.2f} px")

    # Optimize distortion parameters (shared distortion, per-anchor H)
    def cost(params):
        rmses = []
        for pts_src, pts_dst in anchor_point_pairs:
            pts_d = _distort_points(pts_src, params, img_w, img_h)
            H_opt, mask_opt = estimate_homography(
                pts_d.astype(np.float32), pts_dst.astype(np.float32),
            )
            if H_opt is None:
                rmses.append(1e10)
                continue
            rmse, _ = _compute_rmse(H_opt, pts_d, pts_dst, inlier_mask=mask_opt)
            rmses.append(rmse)
        return float(np.mean(rmses))

    try:
        result = scipy_minimize(
            cost,
            x0=np.zeros(8),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-6},
        )
        opt_params = result.x
        opt_rmse = result.fun
    except Exception as e:
        logger.warning(f"Distortion optimization failed: {e}")
        return None, None, None

    # Check if distortion correction improves RMSE
    improvement = base_rmse - opt_rmse
    logger.info(
        f"Distortion estimation: optimized RMSE = {opt_rmse:.2f} px "
        f"(improvement = {improvement:.2f} px)"
    )

    if improvement < 0.1:
        logger.info("Distortion estimation: no significant improvement, skipping")
        return None, None, None

    # Build remap tables (inverse distortion)
    map_x, map_y = np.meshgrid(
        np.arange(img_w, dtype=np.float32),
        np.arange(img_h, dtype=np.float32),
    )
    grid = np.stack([map_x.flatten(), map_y.flatten()], axis=1)

    inv_params = -opt_params
    grid_d = _distort_points(grid, inv_params, img_w, img_h)
    map_x_d = grid_d[:, 0].reshape(img_h, img_w).astype(np.float32)
    map_y_d = grid_d[:, 1].reshape(img_h, img_w).astype(np.float32)

    logger.info("Distortion estimation: remap tables created")
    return opt_params, map_x_d, map_y_d


# =============================================================================
# RMSE computation
# =============================================================================


def _compute_rmse(
    H: np.ndarray,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
) -> tuple[float, int]:
    """Compute reprojection RMSE for a homography.

    When *inlier_mask* is provided (from RANSAC), RMSE is computed only
    over inlier points.  Otherwise RMSE is computed over all points.

    Args:
        H: 3x3 homography matrix.
        pts_src: Source points (N, 2).
        pts_dst: Destination points (N, 2).
        inlier_mask: Boolean array of length N indicating RANSAC inliers.

    Returns:
        (rmse, inlier_count) — RMSE over inliers and the inlier count.
    """
    n = len(pts_src)
    if n == 0:
        return float("inf"), 0

    ones = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([pts_src.astype(np.float64), ones])
    projected = (H @ pts_h.T).T
    w = projected[:, 2:]
    w[w == 0] = 1e-10
    projected_xy = projected[:, :2] / w

    errors = np.sqrt(
        np.sum((projected_xy - pts_dst.astype(np.float64)) ** 2, axis=1)
    )

    if inlier_mask is not None:
        inlier_errors = errors[inlier_mask]
        inlier_count = int(np.sum(inlier_mask))
    else:
        inlier_errors = errors
        inlier_count = n

    if len(inlier_errors) == 0:
        return float("inf"), 0

    rmse = float(np.sqrt(np.mean(inlier_errors**2)))
    return rmse, inlier_count


# =============================================================================
# Match point filtering helpers
# =============================================================================


def _grid_sample_matches(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    grid_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Spatially thin match points using a grid.

    Divides the source image space into a grid of *grid_size* x *grid_size*
    pixel cells and keeps at most one point per cell (the one closest to
    the cell centre).  This ensures even spatial distribution and prevents
    dense clusters from dominating the homography estimation.

    Args:
        pts_src: Source points, shape (N, 2).
        pts_dst: Destination points, shape (N, 2).
        grid_size: Cell size in pixels.

    Returns:
        Filtered ``(pts_src, pts_dst)`` with at most one point per cell.
    """
    if len(pts_src) == 0:
        return pts_src, pts_dst

    # Assign each point to a grid cell
    cell_x = pts_src[:, 0] // grid_size
    cell_y = pts_src[:, 1] // grid_size

    # Compute distance to cell centre for each point
    cx = (cell_x + 0.5) * grid_size
    cy = (cell_y + 0.5) * grid_size
    dist = (pts_src[:, 0] - cx) ** 2 + (pts_src[:, 1] - cy) ** 2

    # Group by cell, keep the closest to centre
    cell_ids = cell_y * 100000 + cell_x  # unique cell id
    keep = np.zeros(len(pts_src), dtype=bool)

    unique_cells = np.unique(cell_ids)
    for cid in unique_cells:
        mask = cell_ids == cid
        indices = np.where(mask)[0]
        best = indices[np.argmin(dist[indices])]
        keep[best] = True

    logger.debug(
        f"Grid sampling: {len(pts_src)} -> {int(np.sum(keep))} "
        f"(grid_size={grid_size})"
    )
    return pts_src[keep], pts_dst[keep]


def _filter_sky_matches(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    sky_mask: Optional[np.ndarray],
    cloud_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Remove match points that fall in sky or cloud regions.

    Points whose source coordinates land on sky or cloud pixels are
    excluded because cloud features are unreliable for alignment.

    Args:
        pts_src: Source points, shape (N, 2).
        pts_dst: Destination points, shape (N, 2).
        sky_mask: Boolean sky mask (H, W), or None.
        cloud_mask: Boolean cloud mask (H, W), or None.

    Returns:
        Filtered ``(pts_src, pts_dst)``.
    """
    if len(pts_src) == 0:
        return pts_src, pts_dst

    if sky_mask is None and cloud_mask is None:
        return pts_src, pts_dst

    keep = np.ones(len(pts_src), dtype=bool)

    for mask in [sky_mask, cloud_mask]:
        if mask is None:
            continue
        h, w = mask.shape[:2]
        x = pts_src[:, 0].astype(int)
        y = pts_src[:, 1].astype(int)
        # Clamp to image bounds
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        keep &= ~mask[y, x]

    n_removed = len(pts_src) - int(np.sum(keep))
    if n_removed > 0:
        logger.debug(
            f"Sky/cloud filter: removed {n_removed}/{len(pts_src)} points"
        )
    return pts_src[keep], pts_dst[keep]


# =============================================================================
# Mask warping helpers
# =============================================================================


def _warp_mask(
    mask: Optional[np.ndarray],
    H: np.ndarray,
    map_x: Optional[np.ndarray],
    map_y: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Warp a boolean SAM mask through distortion correction + homography.

    Args:
        mask: Boolean mask array (H, W), or None.
        H: 3x3 homography matrix.
        map_x: Distortion remap table (x), or None.
        map_y: Distortion remap table (y), or None.

    Returns:
        Warped boolean mask, or None if input is None.
    """
    if mask is None:
        return None

    mask_u8 = mask.astype(np.uint8) * 255
    if map_x is not None and map_y is not None:
        mask_u8 = cv2.remap(
            mask_u8, map_x, map_y, interpolation=cv2.INTER_NEAREST,
        )
    h, w = mask_u8.shape[:2]
    warped = cv2.warpPerspective(mask_u8, H, (w, h), flags=cv2.INTER_NEAREST)
    return warped > 127


def _warp_sam_ridge(
    sam_ridge: Optional[np.ndarray],
    H: np.ndarray,
    map_x: Optional[np.ndarray],
    map_y: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Warp a per-column SAM ridge array through distortion + homography.

    Converts the ridge to a thin binary mask, warps it, then extracts
    the per-column top pixel as the new ridge guide.

    Args:
        sam_ridge: Per-column y coordinate array (int32), or None.
        H: 3x3 homography matrix.
        map_x: Distortion remap table (x), or None.
        map_y: Distortion remap table (y), or None.

    Returns:
        Warped per-column y coordinate array, or None if input is None.
    """
    if sam_ridge is None:
        return None

    w = len(sam_ridge)
    # Determine image height from the ridge values
    valid = sam_ridge[sam_ridge >= 0]
    if len(valid) == 0:
        return None
    h = int(np.max(valid)) + 100  # add margin below the ridge

    # Build a thin mask around the ridge (±2 px band)
    mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        y = sam_ridge[x]
        if 0 <= y < h:
            y_lo = max(0, y - 2)
            y_hi = min(h, y + 3)
            mask[y_lo:y_hi, x] = 255

    if map_x is not None and map_y is not None:
        # Crop remap tables to our mask height
        mx = map_x[:h, :w] if map_x.shape[0] >= h else map_x
        my = map_y[:h, :w] if map_y.shape[0] >= h else map_y
        if mx.shape[0] < h:
            # Pad if necessary
            pad_h = h - mx.shape[0]
            mx = np.pad(mx, ((0, pad_h), (0, 0)), mode="edge")
            my = np.pad(my, ((0, pad_h), (0, 0)), mode="edge")
        mask = cv2.remap(mask, mx, my, interpolation=cv2.INTER_NEAREST)

    warped = cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST)

    # Extract per-column top pixel
    result = np.full(w, -1, dtype=np.int32)
    for x in range(w):
        col = warped[:, x]
        positions = np.where(col > 127)[0]
        if len(positions) > 0:
            result[x] = positions[0]

    return result


# =============================================================================
# Single-segment alignment attempt
# =============================================================================


def _try_align_segment(
    candidates: list[dict],
    ref_path: str,
    method: str,
    device: str,
    resize: Optional[int],
    min_ransac_inliers: int,
    max_rmse: float,
    ridge_validation: bool,
    target_ridge: Optional[np.ndarray],
    masks_dir: Optional[Path],
    map_x: Optional[np.ndarray],
    map_y: Optional[np.ndarray],
    grid_size: int = 100,
    ransac_thresh: float = 10.0,
) -> Optional[dict]:
    """Try aligning a segment using candidate anchors.

    Tries candidates in order, applying 3-layer validation:
      - Layer 1: RANSAC inlier count >= min_ransac_inliers
      - Layer 2: Reprojection RMSE <= max_rmse
      - Layer 3: Ridgeline match on warped image vs target ridgeline

    Before validation, raw matches are filtered:
      - Sky/cloud points removed using anchor SAM masks
      - Spatially thinned via grid sampling

    For Layer 3, the anchor's original SAM masks are warped through
    the same distortion + homography transform so they accurately
    guide ridgeline detection on the warped image.

    Args:
        candidates: Anchor candidates (from _select_anchor_candidates).
        ref_path: Reference image path to match against.
        method: Matching method.
        device: Compute device.
        resize: Resize parameter.
        min_ransac_inliers: Minimum inlier count (Layer 1).
        max_rmse: Maximum RMSE (Layer 2).
        ridge_validation: Enable ridgeline validation (Layer 3).
        target_ridge: Target ridgeline array for Layer 3.
        masks_dir: SAM masks directory for Layer 3.
        map_x: Distortion remap table (x).
        map_y: Distortion remap table (y).
        grid_size: Grid cell size in pixels for spatial thinning.
        ransac_thresh: RANSAC reprojection threshold in pixels for
            homography estimation.

    Returns:
        Tuple of ``(result, best_failed)``.
        *result* is a dict with ``H``, ``rmse``, ``num_matches``, ``anchor``,
        ``pts_src_inlier``, ``pts_dst_inlier`` on success, or None if all
        candidates fail.
        *best_failed* is the best failing attempt dict (same keys plus
        ``fail_reason``), or None if no attempt reached Layer 1.
    """
    from yamami.ridgeline import load_mask

    best_failed: Optional[dict] = None
    best_failed_inliers = -1

    for cand in candidates:
        anchor_path = cand["path"]
        try:
            pts_src, pts_dst = _match_images(
                anchor_path, ref_path, method=method, device=device,
                resize=resize,
            )
        except Exception as e:
            logger.debug(f"Matching failed for {anchor_path}: {e}")
            continue

        n_raw = len(pts_src)

        # Filter out points in sky/cloud regions
        if masks_dir is not None:
            basename = Path(anchor_path).stem
            sky = load_mask(masks_dir / f"{basename}_sky.npz")
            cloud = load_mask(masks_dir / f"{basename}_cloud.npz")
            pts_src, pts_dst = _filter_sky_matches(
                pts_src, pts_dst, sky, cloud,
            )

        # Spatial grid thinning
        if grid_size > 0:
            pts_src, pts_dst = _grid_sample_matches(
                pts_src, pts_dst, grid_size=grid_size,
            )

        logger.info(
            f"Anchor {Path(anchor_path).name}: "
            f"{n_raw} raw -> {len(pts_src)} filtered matches"
        )

        if len(pts_src) < 4:
            logger.debug(
                f"Anchor {Path(anchor_path).name}: "
                f"insufficient matches ({len(pts_src)})"
            )
            continue

        H, inlier_mask = estimate_homography(
            pts_src.astype(np.float32), pts_dst.astype(np.float32),
            ransac_thresh=ransac_thresh,
        )
        if H is None:
            logger.debug(f"Anchor {Path(anchor_path).name}: homography failed")
            continue

        rmse, inlier_count = _compute_rmse(
            H, pts_src, pts_dst, inlier_mask=inlier_mask,
        )

        # Build candidate result for tracking the best failed attempt
        pts_src_inlier = pts_src[inlier_mask] if inlier_mask is not None else pts_src
        pts_dst_inlier = pts_dst[inlier_mask] if inlier_mask is not None else pts_dst
        cand_result = {
            "H": H,
            "rmse": rmse,
            "num_matches": inlier_count,
            "anchor": Path(anchor_path).name,
            "method": method,
            "pts_src_inlier": pts_src_inlier,
            "pts_dst_inlier": pts_dst_inlier,
        }

        # Layer 1: Inlier count
        if inlier_count < min_ransac_inliers:
            logger.debug(
                f"Anchor {Path(anchor_path).name}: "
                f"Layer 1 fail (inliers={inlier_count} < {min_ransac_inliers})"
            )
            if inlier_count > best_failed_inliers:
                best_failed = {**cand_result, "fail_reason": f"Layer1: inliers={inlier_count}"}
                best_failed_inliers = inlier_count
            continue

        # Layer 2: RMSE
        if rmse > max_rmse:
            logger.debug(
                f"Anchor {Path(anchor_path).name}: "
                f"Layer 2 fail (RMSE={rmse:.2f} > {max_rmse})"
            )
            if inlier_count > best_failed_inliers:
                best_failed = {**cand_result, "fail_reason": f"Layer2: RMSE={rmse:.2f}"}
                best_failed_inliers = inlier_count
            continue

        # Layer 3: Ridgeline validation
        if ridge_validation and target_ridge is not None and masks_dir is not None:
            from yamami.ridgeline import (
                detect_ridgeline,
                get_sam_ridge,
                load_mask,
                match_ridgelines_ransac,
            )

            anchor_img = cv2.imread(anchor_path)
            if anchor_img is not None:
                # Apply distortion correction + homography
                if map_x is not None and map_y is not None:
                    anchor_img = cv2.remap(
                        anchor_img, map_x, map_y,
                        interpolation=cv2.INTER_LINEAR,
                    )
                warped = apply_warp(anchor_img, H)

                # Load anchor's SAM masks and warp them through
                # the same transform so they serve as accurate
                # guides in the warped coordinate system.
                basename = Path(anchor_path).stem
                w_sam_ridge = _warp_sam_ridge(
                    get_sam_ridge(masks_dir / f"{basename}_mountain.npz"),
                    H, map_x, map_y,
                )
                w_sky = _warp_mask(
                    load_mask(masks_dir / f"{basename}_sky.npz"),
                    H, map_x, map_y,
                )
                w_cloud = _warp_mask(
                    load_mask(masks_dir / f"{basename}_cloud.npz"),
                    H, map_x, map_y,
                )

                warped_ridge, _, _ = detect_ridgeline(
                    warped, w_sam_ridge, w_sky, w_cloud,
                )
                mr = match_ridgelines_ransac(target_ridge, warped_ridge)

                if not mr["match_success"]:
                    logger.debug(
                        f"Anchor {Path(anchor_path).name}: "
                        f"Layer 3 fail (ridge inlier_ratio={mr['inlier_ratio']:.2f})"
                    )
                    if inlier_count > best_failed_inliers:
                        best_failed = {
                            **cand_result,
                            "fail_reason": f"Layer3: inlier_ratio={mr['inlier_ratio']:.2f}",
                        }
                        best_failed_inliers = inlier_count
                    continue

        logger.info(
            f"Anchor {Path(anchor_path).name}: "
            f"success (RMSE={rmse:.2f}, inliers={inlier_count})"
        )

        return cand_result, None

    return None, best_failed


# =============================================================================
# Round-based alignment
# =============================================================================


def _select_aligned_reference(
    df: pd.DataFrame,
    segment_results: dict,
    target_seg_id: int,
    target_blue_ratio: float,
    aligned_dir: str,
) -> Optional[str]:
    """Select best aligned image as reference for Round 2+.

    Picks from successfully aligned segments the image whose blue_ratio
    is closest to the target segment's median blue_ratio.

    Args:
        df: DataFrame.
        segment_results: Current alignment results.
        target_seg_id: Segment that needs alignment.
        target_blue_ratio: Target blue ratio for scoring.
        aligned_dir: Directory with aligned images.

    Returns:
        Path to aligned reference image, or None.
    """
    best_path = None
    best_score = -1.0

    for seg_id, sr in segment_results.items():
        if sr["status"] != "success":
            continue

        anchor_name = sr.get("anchor", "")
        if not anchor_name:
            continue

        aligned_path = Path(aligned_dir) / anchor_name
        if not aligned_path.exists():
            continue

        # Score by blue_ratio proximity
        seg_df = df[df["segment_id"] == seg_id]
        br_vals = seg_df["ridge_blue_ratio"].dropna()
        br = br_vals.median() if len(br_vals) > 0 else 0.33

        score = 1.0 - min(abs(br - target_blue_ratio) / 0.2, 1.0)
        if score > best_score:
            best_score = score
            best_path = str(aligned_path)

    return best_path


def _round_based_alignment(
    df: pd.DataFrame,
    target_path: str,
    segment_candidates: dict[int, list[dict]],
    method: str,
    device: str,
    resize: Optional[int],
    max_rounds: int,
    min_ransac_inliers: int,
    max_rmse: float,
    ridge_validation: bool,
    target_ridge: Optional[np.ndarray],
    masks_dir: Path,
    aligned_dir: str,
    seg_maps: dict[int, tuple[Optional[np.ndarray], Optional[np.ndarray]]],
    grid_size: int = 100,
    ransac_thresh: float = 10.0,
) -> dict[int, dict]:
    """Iterative round-based alignment.

    Round 1: Each segment tries matching anchor candidates to the target
    image directly.

    Round 2+: Failed segments try matching against aligned images from
    successful segments (which are already in target coordinate system,
    so no transform composition is needed).

    Args:
        seg_maps: Per-segment distortion maps.
            ``{segment_id: (map_x, map_y)}``.  Values may be
            ``(None, None)`` when distortion correction is disabled.

    Terminates when all segments succeed, no new successes occur, or
    max_rounds is reached.

    Returns:
        Dict mapping segment_id to result dict with keys: ``status``,
        ``H``, ``rmse``, ``num_matches``, ``anchor``, ``round``, ``method``.
    """
    segment_results: dict[int, dict] = {}
    pending_segments = set(segment_candidates.keys())

    for round_num in range(1, max_rounds + 1):
        if not pending_segments:
            break

        logger.info(
            f"Round {round_num}: {len(pending_segments)} segments pending"
        )
        new_successes = 0

        for seg_id in list(pending_segments):
            candidates = segment_candidates.get(seg_id, [])
            if not candidates:
                segment_results[seg_id] = {
                    "status": "failed",
                    "H": None,
                    "rmse": np.nan,
                    "num_matches": 0,
                    "anchor": "",
                    "round": -1,
                    "method": method,
                }
                pending_segments.discard(seg_id)
                continue

            # Determine reference for this round
            if round_num == 1:
                ref_path = target_path
            else:
                # Use aligned image from a successful segment
                seg_df = df[df["segment_id"] == seg_id]
                br_vals = seg_df["ridge_blue_ratio"].dropna()
                seg_br = br_vals.median() if len(br_vals) > 0 else 0.33

                ref_path = _select_aligned_reference(
                    df, segment_results, seg_id, seg_br, aligned_dir,
                )
                if ref_path is None:
                    continue  # No reference available yet

            seg_mx, seg_my = seg_maps.get(seg_id, (None, None))

            result, best_failed = _try_align_segment(
                candidates,
                ref_path,
                method=method,
                device=device,
                resize=resize,
                min_ransac_inliers=min_ransac_inliers,
                max_rmse=max_rmse,
                ridge_validation=ridge_validation,
                target_ridge=target_ridge,
                masks_dir=masks_dir,
                map_x=seg_mx,
                map_y=seg_my,
                grid_size=grid_size,
                ransac_thresh=ransac_thresh,
            )

            # Track best failed attempt per segment (for diagnostics)
            if best_failed is not None:
                prev = segment_results.get(seg_id)
                if (
                    prev is None
                    or prev.get("status") != "success"
                    and best_failed["num_matches"] > prev.get("num_matches", 0)
                ):
                    segment_results[seg_id] = {
                        **best_failed,
                        "status": "failed",
                        "round": round_num,
                        "ref_path": ref_path,
                    }

            if result is not None:
                result["status"] = "success"
                result["round"] = round_num
                result["ref_path"] = ref_path
                segment_results[seg_id] = result
                pending_segments.discard(seg_id)
                new_successes += 1

                # Save the anchor's aligned image for use in later rounds
                anchor_name = result["anchor"]
                # Find the anchor path from candidates
                anchor_path = None
                for c in candidates:
                    if Path(c["path"]).name == anchor_name:
                        anchor_path = c["path"]
                        break

                if anchor_path is not None:
                    anchor_img = cv2.imread(anchor_path)
                    if anchor_img is not None:
                        if seg_mx is not None and seg_my is not None:
                            anchor_img = cv2.remap(
                                anchor_img, seg_mx, seg_my,
                                interpolation=cv2.INTER_LINEAR,
                            )
                        warped = apply_warp(anchor_img, result["H"])
                        out_path = Path(aligned_dir) / anchor_name
                        cv2.imwrite(str(out_path), warped)

        logger.info(
            f"Round {round_num}: {new_successes} new successes"
        )

        if new_successes == 0:
            logger.info("No new successes, stopping rounds")
            break

    # Mark remaining pending as failed (keep best_failed data if present)
    for seg_id in pending_segments:
        if seg_id not in segment_results:
            segment_results[seg_id] = {
                "status": "failed",
                "H": None,
                "rmse": np.nan,
                "num_matches": 0,
                "anchor": "",
                "round": -1,
                "method": method,
            }

    return segment_results


# =============================================================================
# Warp segment images
# =============================================================================


def _warp_segment_images(
    df: pd.DataFrame,
    segment_id: int,
    H: np.ndarray,
    map_x: Optional[np.ndarray],
    map_y: Optional[np.ndarray],
    aligned_dir: str,
) -> None:
    """Apply distortion correction + homography to all images in a segment.

    Args:
        df: DataFrame with ``segment_id`` and ``path`` columns.
        segment_id: Segment to process.
        H: 3x3 homography matrix.
        map_x: Distortion remap table (x), or None.
        map_y: Distortion remap table (y), or None.
        aligned_dir: Output directory.
    """
    seg_df = df[df["segment_id"] == segment_id]

    for _, row in seg_df.iterrows():
        filepath = row["path"]
        output_path = Path(aligned_dir) / Path(filepath).name

        if output_path.exists():
            continue  # Already saved (e.g. anchor image)

        img = cv2.imread(filepath)
        if img is None:
            logger.warning(f"Could not read image: {filepath}")
            continue

        if map_x is not None and map_y is not None:
            img = cv2.remap(
                img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            )
        aligned = apply_warp(img, H)
        cv2.imwrite(str(output_path), aligned)


# =============================================================================
# Retained helper functions
# =============================================================================


def estimate_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_thresh: float = 10.0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate homography matrix from matched point pairs.

    Uses RANSAC to robustly estimate the homography transformation.

    Args:
        src_pts: Source points array of shape (N, 2).
        dst_pts: Destination points array of shape (N, 2).
        ransac_thresh: RANSAC reprojection threshold in pixels.

    Returns:
        Tuple of (H, mask) where H is a 3x3 homography matrix and mask
        is a boolean array indicating RANSAC inliers. Both are None if
        estimation fails.
    """
    if len(src_pts) < 4 or len(dst_pts) < 4:
        return None, None

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if H is None:
        return None, None
    inlier_mask = mask.ravel().astype(bool) if mask is not None else None
    return H, inlier_mask


def apply_warp(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply homography transformation to an image.

    Args:
        img: Input image.
        H: 3x3 homography matrix.

    Returns:
        Warped image.
    """
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h))


def _opencv_match(
    img1: np.ndarray,
    img2: np.ndarray,
    detector_type: str = "akaze",
    ratio: float = _DEFAULT_LOWE_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV feature matching using AKAZE or SIFT.

    Args:
        img1: First image (BGR).
        img2: Second image (BGR).
        detector_type: Feature detector type: ``"akaze"`` or ``"sift"``.
        ratio: Lowe's ratio test threshold.

    Returns:
        (pts1, pts2) matched point pairs, shape (N, 2).
    """
    if detector_type.lower() == "akaze":
        detector = cv2.AKAZE_create()
    elif detector_type.lower() == "sift":
        detector = cv2.SIFT_create()
    else:
        raise ValueError(f"Unknown detector_type: {detector_type}")

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return (
            np.array([]).reshape(0, 2).astype("int32"),
            np.array([]).reshape(0, 2).astype("int32"),
        )

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio * n.distance:
                good.append(m)

    if len(good) == 0:
        return (
            np.array([]).reshape(0, 2).astype("int32"),
            np.array([]).reshape(0, 2).astype("int32"),
        )

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).astype("int32")
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).astype("int32")

    return pts1, pts2


def _imm_match(
    path1: str,
    path2: str,
    method: str,
    device: str = "cpu",
    resize: Optional[int] = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matching using imm (image-matching-models) package.

    Args:
        path1: Path to first image.
        path2: Path to second image.
        method: IMM matching method name.
        device: Device for computation.
        resize: Resize images to this size (optional).
        **kwargs: Additional arguments for imm's get_matcher.

    Returns:
        (pts1, pts2) matched point pairs, shape (N, 2).

    Raises:
        ImportError: If imm package is not installed.
    """
    try:
        from imm import get_matcher
    except ImportError:
        raise ImportError(
            f"Method '{method}' requires the 'imm' package. "
            "Install with: pip install image-matching-models"
        )

    img1_cv = cv2.imread(path1)
    img2_cv = cv2.imread(path2)

    if img1_cv is None:
        raise ValueError(f"Could not read image: {path1}")
    if img2_cv is None:
        raise ValueError(f"Could not read image: {path2}")

    h1_org, w1_org = img1_cv.shape[:2]
    h2_org, w2_org = img2_cv.shape[:2]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        matcher = get_matcher(method, device=device, **kwargs)

    img0 = matcher.load_image(path1, resize=resize)
    img1 = matcher.load_image(path2, resize=resize)

    if img0.dim() == 4:
        _, _, h0_loaded, w0_loaded = img0.shape
        _, _, h1_loaded, w1_loaded = img1.shape
    else:
        _, h0_loaded, w0_loaded = img0.shape
        _, h1_loaded, w1_loaded = img1.shape

    result = matcher(img0, img1)
    pts1 = result["matched_kpts0"]
    pts2 = result["matched_kpts1"]

    if hasattr(pts1, "cpu"):
        pts1 = pts1.cpu().numpy()
        pts2 = pts2.cpu().numpy()

    if len(pts1) == 0:
        return (
            np.array([]).reshape(0, 2).astype("int32"),
            np.array([]).reshape(0, 2).astype("int32"),
        )

    scale_x0 = w1_org / w0_loaded
    scale_y0 = h1_org / h0_loaded
    scale_x1 = w2_org / w1_loaded
    scale_y1 = h2_org / h1_loaded

    pts1[:, 0] *= scale_x0
    pts1[:, 1] *= scale_y0
    pts2[:, 0] *= scale_x1
    pts2[:, 1] *= scale_y1

    pts1 = pts1.astype("int32")
    pts2 = pts2.astype("int32")

    return pts1, pts2


def _match_images(
    path1: str,
    path2: str,
    method: str = "akaze",
    device: str = "cpu",
    resize: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match two images using specified method.

    Dispatches to appropriate matching function based on method name.

    Args:
        path1: Path to first image (anchor).
        path2: Path to second image (reference).
        method: Matching method.
        device: Device for IMM methods.
        resize: Resize for IMM methods.

    Returns:
        (pts1, pts2) matched point pairs in original coordinates, shape (N, 2).
    """
    method_lower = method.lower()

    if method_lower in ("akaze", "sift"):
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None:
            logger.warning(f"Could not read image: {path1}")
            return (
                np.array([]).reshape(0, 2).astype("int32"),
                np.array([]).reshape(0, 2).astype("int32"),
            )
        if img2 is None:
            logger.warning(f"Could not read image: {path2}")
            return (
                np.array([]).reshape(0, 2).astype("int32"),
                np.array([]).reshape(0, 2).astype("int32"),
            )

        return _opencv_match(img1, img2, detector_type=method_lower)

    elif method_lower in [m.lower() for m in IMM_METHODS]:
        effective_resize = resize

        if method_lower not in [m.lower() for m in _LIGHTGLUE_METHODS]:
            if resize is None:
                effective_resize = _DEFAULT_RESIZE_HEAVY
                logger.debug(
                    f"Method '{method}' is memory-intensive. "
                    f"Resizing images to {_DEFAULT_RESIZE_HEAVY}px."
                )

        return _imm_match(path1, path2, method, device, resize=effective_resize)
    else:
        available = ["akaze", "sift"] + IMM_METHODS
        raise ValueError(f"Unknown method '{method}'. Available methods: {available}")
