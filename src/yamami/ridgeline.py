"""
Ridgeline detection, matching, and segmentation module for yamami.

Provides adaptive ridgeline detection (Sky Index / Blue Gradient),
RANSAC-based ridgeline matching with affine transform estimation,
and camera shift segmentation based on corner displacement.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize


# =============================================================================
# Blue Ratio
# =============================================================================


def compute_blue_ratio(
    img: np.ndarray, sky_mask: np.ndarray, cloud_mask: np.ndarray
) -> float:
    """Compute blue ratio = mean(B / (R+G+B)) over sky+cloud region.

    Args:
        img: BGR image.
        sky_mask: Boolean mask for sky region (or None).
        cloud_mask: Boolean mask for cloud region (or None).

    Returns:
        Blue ratio value. Returns 0.33 if combined mask has < 100 pixels.
    """
    combined = None
    if sky_mask is not None and cloud_mask is not None:
        combined = sky_mask | cloud_mask
    elif sky_mask is not None:
        combined = sky_mask
    elif cloud_mask is not None:
        combined = cloud_mask

    if combined is None or np.sum(combined) < 100:
        return 0.33

    b = img[:, :, 0][combined].astype(np.float32)
    g = img[:, :, 1][combined].astype(np.float32)
    r = img[:, :, 2][combined].astype(np.float32)
    return float(np.mean(b / (r + g + b + 1e-6)))


# =============================================================================
# Ridgeline detection
# =============================================================================


def _search_bounds(sam_ridge, x, h, search_margin):
    """Return (start, end) search range for column *x*."""
    if sam_ridge is not None and sam_ridge[x] >= 0:
        return max(0, sam_ridge[x] - search_margin), min(h, sam_ridge[x] + search_margin)
    return 0, h // 2


def _find_ridge_from_gradient(
    grad_y: np.ndarray, sam_ridge: np.ndarray, search_margin: int = 10
) -> np.ndarray:
    """Find ridge as argmin of vertical gradient per column."""
    h, w = grad_y.shape
    ridgeline = np.full(w, -1, dtype=np.int32)

    for x in range(w):
        s, e = _search_bounds(sam_ridge, x, h, search_margin)
        col = grad_y[s:e, x]
        if len(col) > 0:
            ridgeline[x] = s + np.argmin(col)

    return ridgeline




def method_sky_index(
    img: np.ndarray, sam_ridge: np.ndarray, search_margin: int = 10
) -> np.ndarray:
    """Sky Index method: gradient of (2B - R - G)."""
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    grad_y = cv2.Sobel(2 * b - r - g, cv2.CV_32F, 0, 1, ksize=5)
    return _find_ridge_from_gradient(grad_y, sam_ridge, search_margin)


def method_blue_gradient(
    img: np.ndarray, sam_ridge: np.ndarray, search_margin: int = 10
) -> np.ndarray:
    """Blue gradient method: negative peak of dB/dy."""
    grad_y = cv2.Sobel(
        img[:, :, 0].astype(np.float32), cv2.CV_32F, 0, 1, ksize=5
    )
    return _find_ridge_from_gradient(grad_y, sam_ridge, search_margin)




def detect_ridgeline(
    img: np.ndarray,
    sam_ridge: np.ndarray,
    sky_mask: np.ndarray,
    cloud_mask: np.ndarray,
    search_margin: int = 10,
    blue_ratio_threshold: float = 0.34,
) -> tuple:
    """Adaptive ridgeline detection.

    Method selection:
        blue_ratio > threshold    -> Sky Index
        otherwise                 -> Blue Gradient

    Args:
        img: BGR image.
        sam_ridge: Approximate ridge from SAM mountain mask (per-column y).
        sky_mask: Boolean sky mask (or None).
        cloud_mask: Boolean cloud mask (or None).
        search_margin: Search range around SAM ridge (pixels).
        blue_ratio_threshold: Threshold to switch between methods.

    Returns:
        Tuple of (ridgeline, method_name, blue_ratio).
    """
    blue_ratio = compute_blue_ratio(img, sky_mask, cloud_mask)

    if blue_ratio > blue_ratio_threshold:
        ridge = method_sky_index(img, sam_ridge, search_margin)
        method_name = "sky_index"
    else:
        ridge = method_blue_gradient(img, sam_ridge, search_margin)
        method_name = "blue_grad"

    return ridge, method_name, blue_ratio


# =============================================================================
# Matching (RANSAC)
# =============================================================================


def match_ridgelines_ransac(
    ridge_target: np.ndarray,
    ridge_query: np.ndarray,
    margin_ratio: float = 0.05,
    inlier_threshold: float = 3.0,
    inlier_ratio_threshold: float = 0.8,
    n_iterations: int = 50,
    sample_ratio: float = 0.3,
) -> dict:
    """RANSAC-based ridgeline matching with affine transform.

    1. Random sample -> parameter estimation
    2. Compute residuals on all points, count inliers
    3. Keep best parameters
    4. Final estimation using inliers only

    Args:
        ridge_target: Target (reference) ridgeline array.
        ridge_query: Query ridgeline array.
        margin_ratio: Fraction of image width to exclude at edges.
        inlier_threshold: Max residual (px) to count as inlier.
        inlier_ratio_threshold: Min inlier ratio for match success.
        n_iterations: Number of RANSAC iterations.
        sample_ratio: Fraction of points to sample per iteration.

    Returns:
        Dict with keys: dx, dy, angle, scale, error_median, error_mean,
        n_matched, n_inliers, inlier_ratio, match_success, residual_mask.
    """
    w = len(ridge_target)
    h_approx = (
        np.max(ridge_target[ridge_target >= 0])
        if np.any(ridge_target >= 0)
        else 500
    )

    margin_px = int(w * margin_ratio)

    valid_target = ridge_target >= 0
    valid_query = ridge_query >= 0

    empty_result = {
        "dx": 0.0,
        "dy": 0.0,
        "angle": 0.0,
        "scale": 1.0,
        "error_median": float("inf"),
        "error_mean": float("inf"),
        "n_matched": 0,
        "n_inliers": 0,
        "inlier_ratio": 0.0,
        "match_success": False,
        "residual_mask": np.full(w, np.nan, dtype=np.float32),
    }

    if np.sum(valid_target) < 100 or np.sum(valid_query) < 100:
        return empty_result

    # Interpolation function for target ridgeline
    x_target = np.where(valid_target)[0]
    y_target = ridge_target[valid_target]
    target_interp = interp1d(
        x_target, y_target, kind="linear", bounds_error=False, fill_value=np.nan
    )

    # Query point cloud
    x_query = np.where(valid_query)[0].astype(np.float64)
    y_query = ridge_query[valid_query].astype(np.float64)

    # Transform center
    cx, cy = w / 2, h_approx / 2

    def transform_points(params, x, y):
        dx, dy, angle_deg, scale = params
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x_centered = x - cx
        y_centered = y - cy
        x_rot = scale * (cos_a * x_centered - sin_a * y_centered)
        y_rot = scale * (sin_a * x_centered + cos_a * y_centered)
        return x_rot + cx + dx, y_rot + cy + dy

    def compute_residuals(params, x, y):
        x_transformed, y_transformed = transform_points(params, x, y)
        y_target_interp = target_interp(x_transformed)
        valid = ~np.isnan(y_target_interp)
        valid &= (x_transformed >= margin_px) & (x_transformed <= w - margin_px)
        residuals = np.full(len(x), np.inf)
        if np.sum(valid) > 0:
            residuals[valid] = np.abs(y_transformed[valid] - y_target_interp[valid])
        return residuals, valid

    def fit_params(x, y, initial_params):
        def cost(params):
            x_t, y_t = transform_points(params, x, y)
            y_ti = target_interp(x_t)
            v = ~np.isnan(y_ti)
            v &= (x_t >= margin_px) & (x_t <= w - margin_px)
            if np.sum(v) < 10:
                return 1e10
            return np.sum((y_t[v] - y_ti[v]) ** 2)

        result = minimize(
            cost,
            initial_params,
            method="L-BFGS-B",
            bounds=[(-100, 100), (-100, 100), (-5, 5), (0.95, 1.05)],
            options={"maxiter": 100},
        )
        return result.x

    # RANSAC
    n_points = len(x_query)
    sample_size = max(50, int(n_points * sample_ratio))

    best_params = [0.0, 0.0, 0.0, 1.0]
    best_n_inliers = 0

    for _ in range(n_iterations):
        indices = np.random.choice(n_points, sample_size, replace=False)
        x_sample = x_query[indices]
        y_sample = y_query[indices]

        try:
            params = fit_params(x_sample, y_sample, best_params)
        except Exception:
            continue

        residuals, valid = compute_residuals(params, x_query, y_query)
        inliers = (residuals <= inlier_threshold) & valid
        n_inliers = np.sum(inliers)

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_params = params

    # Final estimation with inliers only
    residuals, valid = compute_residuals(best_params, x_query, y_query)
    inlier_mask = (residuals <= inlier_threshold) & valid

    if np.sum(inlier_mask) >= 50:
        final_params = fit_params(
            x_query[inlier_mask], y_query[inlier_mask], best_params
        )
    else:
        final_params = best_params

    # Final residuals
    residuals, valid = compute_residuals(final_params, x_query, y_query)
    inlier_mask = (residuals <= inlier_threshold) & valid

    dx, dy, angle, scale = final_params
    n_matched = int(np.sum(valid))
    n_inliers = int(np.sum(inlier_mask))

    if n_matched == 0:
        return empty_result

    inlier_ratio = n_inliers / n_matched
    inlier_residuals = residuals[inlier_mask]
    error_median = (
        float(np.median(inlier_residuals)) if n_inliers > 0 else float("inf")
    )
    error_mean = (
        float(np.mean(inlier_residuals)) if n_inliers > 0 else float("inf")
    )

    match_success = inlier_ratio >= inlier_ratio_threshold

    # Build residual mask
    residual_mask = np.full(w, np.nan, dtype=np.float32)
    x_query_int = x_query.astype(int)
    for i, x in enumerate(x_query_int):
        if valid[i]:
            residual_mask[x] = residuals[i]

    return {
        "dx": float(dx),
        "dy": float(dy),
        "angle": float(angle),
        "scale": float(scale),
        "error_median": error_median,
        "error_mean": error_mean,
        "n_matched": n_matched,
        "n_inliers": n_inliers,
        "inlier_ratio": float(inlier_ratio),
        "match_success": bool(match_success),
        "residual_mask": residual_mask,
    }


# =============================================================================
# Segment detection
# =============================================================================


def _apply_affine(x, y, params, cx, cy):
    """Apply affine transform to a single point."""
    dx, dy, angle_deg, scale = params
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x_c, y_c = x - cx, y - cy
    x_new = scale * (cos_a * x_c - sin_a * y_c) + cx + dx
    y_new = scale * (sin_a * x_c + cos_a * y_c) + cy + dy
    return x_new, y_new


def max_corner_displacement(params_a, params_b, width, height, cy_override=None):
    """Compute max pixel displacement between two affine transforms at image corners.

    Args:
        params_a: (dx, dy, angle, scale) for transform A.
        params_b: (dx, dy, angle, scale) for transform B.
        width: Image width.
        height: Image height.
        cy_override: Override for transform center cy (e.g. h_approx/2 from RANSAC).

    Returns:
        Maximum displacement in pixels across the four corners.
    """
    cx = width / 2
    cy = cy_override if cy_override is not None else height / 2
    corners = [(0, 0), (width, 0), (0, height), (width, height)]
    max_d = 0.0
    for x, y in corners:
        xa, ya = _apply_affine(x, y, params_a, cx, cy)
        xb, yb = _apply_affine(x, y, params_b, cx, cy)
        d = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
        max_d = max(max_d, d)
    return max_d


def detect_segments(results, width, height, ridge_cy, displacement_threshold=5.0):
    """Assign segment IDs to all images based on camera shift events.

    Places segment boundaries where the max corner displacement between
    consecutive successfully matched images exceeds the threshold.
    Failed/skipped images belong to the same segment until the next shift.

    Also writes ``segment_id``, ``ridge_confirmed``, and
    ``prev_displacement`` into each element of *results* in-place.

    Args:
        results: List of dicts with at least ``match_success``, ``dx``,
            ``dy``, ``angle``, ``scale`` keys.
        width: Image width.
        height: Image height.
        ridge_cy: Transform center cy used in RANSAC (= h_approx / 2).
        displacement_threshold: Pixel threshold for segment boundary.

    Returns:
        List of segment info dicts with keys: segment_id, n_total,
        n_confirmed, dx_median, dy_median, angle_median, scale_median.
    """
    matched_indices = [i for i, r in enumerate(results) if r["match_success"]]

    boundary_set = set()
    for k in range(len(matched_indices)):
        i_curr = matched_indices[k]
        if k == 0:
            results[i_curr]["prev_displacement"] = 0.0
            continue

        i_prev = matched_indices[k - 1]
        r_prev, r_curr = results[i_prev], results[i_curr]

        params_prev = (r_prev["dx"], r_prev["dy"], r_prev["angle"], r_prev["scale"])
        params_curr = (r_curr["dx"], r_curr["dy"], r_curr["angle"], r_curr["scale"])
        disp = max_corner_displacement(
            params_prev, params_curr, width, height, cy_override=ridge_cy
        )

        r_curr["prev_displacement"] = float(disp)

        if disp > displacement_threshold:
            boundary_set.add(i_curr)

    # Assign segment IDs
    seg_id = 0
    for i, r in enumerate(results):
        if i in boundary_set:
            seg_id += 1
        r["segment_id"] = seg_id
        r["ridge_confirmed"] = r["match_success"]

    # Aggregate per segment
    n_segments = seg_id + 1
    segment_info = []
    for sid in range(n_segments):
        seg_all = [r for r in results if r["segment_id"] == sid]
        seg_confirmed = [r for r in seg_all if r["ridge_confirmed"]]

        info = {
            "segment_id": sid,
            "n_total": len(seg_all),
            "n_confirmed": len(seg_confirmed),
        }

        if seg_confirmed:
            dx_vals = [r["dx"] for r in seg_confirmed]
            dy_vals = [r["dy"] for r in seg_confirmed]
            angle_vals = [r["angle"] for r in seg_confirmed]
            scale_vals = [r["scale"] for r in seg_confirmed]
            info["dx_median"] = float(np.median(dx_vals))
            info["dy_median"] = float(np.median(dy_vals))
            info["angle_median"] = float(np.median(angle_vals))
            info["scale_median"] = float(np.median(scale_vals))
        else:
            info["dx_median"] = np.nan
            info["dy_median"] = np.nan
            info["angle_median"] = np.nan
            info["scale_median"] = np.nan

        segment_info.append(info)

    return segment_info


# =============================================================================
# Mask / SAM helpers
# =============================================================================


def get_sam_ridge(mask_path, snow_mask_path=None):
    """Extract approximate ridgeline from SAM mountain mask.

    When *snow_mask_path* is provided, the snow mask is combined with
    the mountain mask before extracting the ridge.  This prevents the
    ridge from being lost when SAM classifies snow-covered mountain
    pixels as "snow" instead of "mountain".

    Args:
        mask_path: Path to mountain .npz file containing ``mask`` array.
        snow_mask_path: Optional path to snow .npz file.

    Returns:
        Per-column y coordinate array (int32), or None if file missing.
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        return None

    mask = np.load(mask_path)["mask"]

    if snow_mask_path is not None:
        snow_path = Path(snow_mask_path)
        if snow_path.exists():
            snow_mask = np.load(snow_path)["mask"]
            if snow_mask.shape == mask.shape:
                mask = mask | snow_mask

    h, w = mask.shape
    ridge = np.full(w, -1, dtype=np.int32)

    for x in range(w):
        col = mask[:, x]
        positions = np.where(col)[0]
        if len(positions) > 0:
            ridge[x] = positions[0]

    return ridge


def load_mask(mask_path):
    """Load a boolean mask from .npz file.

    Args:
        mask_path: Path to .npz file containing ``mask`` array.

    Returns:
        Mask array, or None if file missing.
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        return None
    return np.load(mask_path)["mask"]
