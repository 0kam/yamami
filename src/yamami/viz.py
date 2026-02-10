"""
Visualization module for YAMAMI.

Provides color-coded DOY (Day of Year) visualizations for phenology
and snowmelt data, plus ridgeline overlay and grid image utilities.
"""

import os
from typing import Optional, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use non-interactive backend for server environments
matplotlib.use("Agg")


def viz(
    data: Union[pd.DataFrame, dict],
    colormap: str = "viridis",
    vmin: Optional[int] = None,
    vmax: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Generate color-coded DOY visualizations.

    Creates matplotlib figures with colorbar/legend for phenology or
    snowmelt data. For phenology data (gup, gdown, gmax), generates
    three maps. For snowmelt data (doy), generates a single doy_map.

    Args:
        data: Input data as DataFrame or dict.
            For DataFrame: columns 'row', 'col', and data columns
            (gup, gdown, gmax for phenology; doy for snowmelt).
            For dict: keys are data names, values are 2D numpy arrays.
        colormap: Matplotlib colormap name (default: "viridis")
        vmin: Minimum value for colormap normalization
        vmax: Maximum value for colormap normalization
        output_dir: Directory to save PNG files. If None, returns arrays.

    Returns:
        dict: If output_dir is provided, returns dict of file paths.
              Otherwise, returns dict of numpy arrays (RGBA images).

    Example:
        >>> data = pd.DataFrame({
        ...     "row": [0, 0, 1, 1],
        ...     "col": [0, 1, 0, 1],
        ...     "gup": [100, 110, 105, 115],
        ...     "gdown": [250, 260, 255, 265],
        ...     "gmax": [175, 185, 180, 190],
        ... })
        >>> result = viz(data, output_dir="/path/to/output")
        >>> print(result.keys())
        dict_keys(['gup', 'gdown', 'gmax'])
    """
    # Determine data type and extract arrays
    if isinstance(data, dict):
        arrays = data
    else:
        arrays = _dataframe_to_arrays(data)

    # Determine what type of data we have
    data_keys = list(arrays.keys())
    is_phenology = any(k in data_keys for k in ["gup", "gdown", "gmax"])
    is_snowmelt = "doy" in data_keys or "doy_map" in data_keys

    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    result = {}

    if is_phenology:
        # Generate phenology visualizations
        for key in ["gup", "gdown", "gmax"]:
            if key in arrays:
                array = arrays[key]
                output = _create_visualization(
                    array,
                    title=_get_title(key),
                    colormap=colormap,
                    vmin=vmin,
                    vmax=vmax,
                    output_dir=output_dir,
                    filename=f"{key}.png",
                )
                result[key] = output

    if is_snowmelt:
        # Generate snowmelt visualization
        doy_key = "doy" if "doy" in arrays else "doy_map"
        array = arrays[doy_key]
        output = _create_visualization(
            array,
            title="Snowmelt DOY",
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            output_dir=output_dir,
            filename="doy_map.png",
        )
        result["doy_map"] = output

    return result


def _dataframe_to_arrays(df: pd.DataFrame) -> dict:
    """
    Convert DataFrame with row/col indices to 2D arrays.

    Args:
        df: DataFrame with 'row', 'col' columns and data columns

    Returns:
        dict: Data column names mapped to 2D numpy arrays
    """
    if "row" not in df.columns or "col" not in df.columns:
        raise ValueError("DataFrame must have 'row' and 'col' columns")

    rows = df["row"].values
    cols = df["col"].values

    # Determine array dimensions
    max_row = int(rows.max()) + 1
    max_col = int(cols.max()) + 1

    # Get data columns (exclude row/col)
    data_cols = [c for c in df.columns if c not in ["row", "col"]]

    result = {}
    for col_name in data_cols:
        # Create array filled with NaN
        array = np.full((max_row, max_col), np.nan)

        # Fill in values
        valid_mask = ~df[col_name].isna()
        valid_rows = rows[valid_mask].astype(int)
        valid_cols = cols[valid_mask].astype(int)
        array[valid_rows, valid_cols] = df.loc[valid_mask, col_name].values

        result[col_name] = array

    return result


def _get_title(key: str) -> str:
    """Get display title for data key."""
    titles = {
        "gup": "Green-up DOY",
        "gdown": "Green-down DOY",
        "gmax": "Maximum Greenness DOY",
    }
    return titles.get(key, key.upper())


def _create_visualization(
    array: np.ndarray,
    title: str,
    colormap: str,
    vmin: Optional[int],
    vmax: Optional[int],
    output_dir: Optional[str],
    filename: str,
) -> Union[str, np.ndarray]:
    """
    Create a single visualization with colorbar.

    Args:
        array: 2D numpy array of data
        title: Plot title
        colormap: Matplotlib colormap name
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        output_dir: Output directory (None for array output)
        filename: Output filename

    Returns:
        File path if output_dir provided, otherwise RGBA array
    """
    # Create figure with colorbar
    fig, ax = plt.subplots(figsize=(8, 6))

    # Handle vmin/vmax
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)

    # Create colormap with masked array for NaN handling
    masked_array = np.ma.masked_invalid(array)

    # Plot image
    im = ax.imshow(masked_array, cmap=colormap, vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Day of Year (DOY)")

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    if output_dir is not None:
        # Save to file
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path
    else:
        # Return as array
        fig.canvas.draw()
        # Get RGBA array from figure
        width, height = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape(height, width, 4)
        plt.close(fig)
        return rgba.copy()


# =============================================================================
# Ridgeline visualization
# =============================================================================


def draw_ridgeline(
    img: np.ndarray,
    ridgeline: np.ndarray,
    color=(0, 255, 0),
    thickness=3,
    inlier_mask: np.ndarray = None,
    inlier_color=(0, 255, 0),
    outlier_color=(0, 0, 255),
    residual_mask: np.ndarray = None,
    residual_threshold: float = 5.0,
) -> np.ndarray:
    """Draw ridgeline on image with optional inlier/outlier coloring.

    Args:
        img: BGR image.
        ridgeline: Per-column y coordinate array.
        color: Default line color (BGR).
        thickness: Line thickness.
        inlier_mask: Boolean array indicating inlier columns.
        inlier_color: Color for inlier segments (BGR).
        outlier_color: Color for outlier segments (BGR).
        residual_mask: Per-column residual values (NaN = no match).
            When provided, columns with residual <= residual_threshold
            are drawn as inliers, others as outliers.
        residual_threshold: Threshold for residual-based coloring.

    Returns:
        Copy of image with ridgeline drawn.
    """
    vis = img.copy()
    h, w = img.shape[:2]

    for x in range(0, w - 1, 2):
        y1 = ridgeline[x]
        y2 = ridgeline[x + 1]
        if 0 <= y1 < h and 0 <= y2 < h:
            if residual_mask is not None:
                res = residual_mask[x]
                if np.isnan(res):
                    c = (128, 128, 128)
                elif res <= residual_threshold:
                    c = inlier_color
                else:
                    c = outlier_color
            elif inlier_mask is not None:
                c = inlier_color if inlier_mask[x] else outlier_color
            else:
                c = color
            cv2.line(vis, (x, y1), (x + 1, y2), c, thickness)

    return vis


def create_grid_image(images: list, cols: int = 7) -> np.ndarray:
    """Create a grid image from a list of images.

    Args:
        images: List of BGR images (all same size).
        cols: Number of columns in the grid.

    Returns:
        Grid image as numpy array.
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    rows = (len(images) + cols - 1) // cols

    while len(images) < rows * cols:
        images.append(np.zeros_like(images[0]))

    img_h, img_w = images[0].shape[:2]
    grid = np.zeros((rows * img_h, cols * img_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        r, c = i // cols, i % cols
        grid[r * img_h : (r + 1) * img_h, c * img_w : (c + 1) * img_w] = img

    return grid


# =============================================================================
# Match result visualization
# =============================================================================


def _error_color(error: float, max_err: float) -> tuple:
    """Map reprojection error to BGR color (green -> yellow -> red)."""
    t = min(error / max_err, 1.0)
    if t < 0.5:
        r, g = int(255 * (t * 2)), 255
    else:
        r, g = 255, int(255 * (1.0 - (t - 0.5) * 2))
    return (0, g, r)


def _draw_displacement_arrows(
    img: np.ndarray,
    pts_from: np.ndarray,
    pts_to: np.ndarray,
    errors: np.ndarray,
    max_err: float,
    arrow_scale: float = 5.0,
    thickness: int = 2,
    tip_length: float = 0.3,
) -> np.ndarray:
    """Draw displacement arrows on an image.

    Each arrow starts at ``pts_from[i]`` and points toward ``pts_to[i]``.
    Arrow length is magnified by *arrow_scale* for visibility.
    Color encodes reprojection error (green=low, red=high).

    Args:
        img: BGR image (modified in-place and returned).
        pts_from: Origin points, shape (N, 2), int pixel coords.
        pts_to: Target points, shape (N, 2), int pixel coords.
        errors: Per-point reprojection error array.
        max_err: Error value mapped to pure red.
        arrow_scale: Displacement magnification factor.
        thickness: Arrow line thickness.
        tip_length: Arrow tip length as fraction of arrow length.

    Returns:
        Image with arrows drawn.
    """
    for i in range(len(pts_from)):
        dx = float(pts_to[i, 0] - pts_from[i, 0]) * arrow_scale
        dy = float(pts_to[i, 1] - pts_from[i, 1]) * arrow_scale
        p1 = (int(pts_from[i, 0]), int(pts_from[i, 1]))
        p2 = (int(pts_from[i, 0] + dx), int(pts_from[i, 1] + dy))
        color = _error_color(errors[i], max_err)
        cv2.arrowedLine(
            img, p1, p2, color, thickness, cv2.LINE_AA, tipLength=tip_length,
        )
    return img


def plot_match_result(
    anchor_path: str,
    ref_path: str,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    H: np.ndarray,
    segment_id: int,
    round_num: int,
    rmse: float,
    num_matches: int,
    method: str,
    output_path: Optional[str] = None,
    max_display_width: int = 1200,
    arrow_scale: float = 5.0,
    status_tag: str = "",
    fail_reason: str = "",
) -> np.ndarray:
    """Visualize feature matching result between anchor and reference images.

    Creates a side-by-side image with displacement arrows drawn on each
    panel.  On the anchor (left), arrows point from source toward
    destination coordinates; on the reference (right), arrows point from
    destination toward source.  Arrow length is magnified by *arrow_scale*
    for readability.  Color encodes per-point reprojection error
    (green = low, red = high).

    Args:
        anchor_path: Path to the anchor (source) image.
        ref_path: Path to the reference (destination) image.
        pts_src: Inlier source points, shape (N, 2).
        pts_dst: Inlier destination points, shape (N, 2).
        H: 3x3 homography matrix.
        segment_id: Segment identifier.
        round_num: Alignment round number.
        rmse: Reprojection RMSE (px).
        num_matches: Number of RANSAC inliers.
        method: Matching method name.
        output_path: If provided, save the visualization to this path.
        max_display_width: Maximum width for each image panel.
        arrow_scale: Displacement magnification factor for arrows.
        status_tag: Optional status label (e.g. ``"FAILED"``) shown in
            the top bar.  When ``"FAILED"``, the label is drawn in red.
        fail_reason: Optional failure reason string appended after the
            status tag.

    Returns:
        BGR image of the visualization.
    """
    anchor_img = cv2.imread(anchor_path)
    ref_img = cv2.imread(ref_path)

    if anchor_img is None or ref_img is None:
        blank = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(
            blank, "Could not load images", (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        return blank

    # Resize for display
    h1, w1 = anchor_img.shape[:2]
    h2, w2 = ref_img.shape[:2]
    scale = min(1.0, max_display_width / max(w1, w2))
    if scale < 1.0:
        anchor_disp = cv2.resize(
            anchor_img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        ref_disp = cv2.resize(
            ref_img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        pts_src_s = (pts_src * scale).astype(np.int32)
        pts_dst_s = (pts_dst * scale).astype(np.int32)
    else:
        anchor_disp = anchor_img.copy()
        ref_disp = ref_img.copy()
        pts_src_s = pts_src.astype(np.int32)
        pts_dst_s = pts_dst.astype(np.int32)

    dh1, dw1 = anchor_disp.shape[:2]
    dh2, dw2 = ref_disp.shape[:2]

    # Make both panels same height
    max_h = max(dh1, dh2)
    if dh1 < max_h:
        pad = np.zeros((max_h - dh1, dw1, 3), dtype=np.uint8)
        anchor_disp = np.vstack([anchor_disp, pad])
    if dh2 < max_h:
        pad = np.zeros((max_h - dh2, dw2, 3), dtype=np.uint8)
        ref_disp = np.vstack([ref_disp, pad])

    # Compute per-point reprojection error for color coding
    n = len(pts_src)
    if n > 0:
        ones = np.ones((n, 1), dtype=np.float64)
        pts_h = np.hstack([pts_src.astype(np.float64), ones])
        projected = (H @ pts_h.T).T
        w = projected[:, 2:]
        w[w == 0] = 1e-10
        projected_xy = projected[:, :2] / w
        errors = np.sqrt(
            np.sum((projected_xy - pts_dst.astype(np.float64)) ** 2, axis=1)
        )
    else:
        errors = np.array([])

    # Draw displacement arrows on each image
    max_err = max(rmse * 2.0, 1.0) if rmse > 0 else 5.0
    _draw_displacement_arrows(
        anchor_disp, pts_src_s, pts_dst_s, errors, max_err,
        arrow_scale=arrow_scale,
    )
    _draw_displacement_arrows(
        ref_disp, pts_dst_s, pts_src_s, errors, max_err,
        arrow_scale=arrow_scale,
    )

    # Assemble canvas: top bar + images + bottom bar
    top_bar_h = 40
    bottom_bar_h = 30
    total_w = dw1 + dw2
    canvas = np.zeros(
        (top_bar_h + max_h + bottom_bar_h, total_w, 3), dtype=np.uint8,
    )
    canvas[top_bar_h : top_bar_h + max_h, :dw1] = anchor_disp
    canvas[top_bar_h : top_bar_h + max_h, dw1 : dw1 + dw2] = ref_disp

    # Top bar: stats
    anchor_name = os.path.basename(anchor_path)
    ref_name = os.path.basename(ref_path)
    text = (
        f"Seg {segment_id} | Round {round_num} | "
        f"RMSE {rmse:.2f}px | {num_matches} inliers | {method}"
    )
    if status_tag:
        text = f"[{status_tag}] {text}"
        if fail_reason:
            text += f" | {fail_reason}"
    text_color = (0, 0, 255) if status_tag == "FAILED" else (255, 255, 255)
    cv2.putText(
        canvas, text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA,
    )

    # Bottom bar: image labels
    label_y = top_bar_h + max_h + 22
    cv2.putText(
        canvas, f"Anchor: {anchor_name}", (10, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"Ref: {ref_name}", (dw1 + 10, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
    )

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, canvas)

    return canvas
