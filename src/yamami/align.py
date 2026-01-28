"""
Image alignment module for yamami.

Provides functions for detecting camera shifts, segmenting time series
into stable intervals, and aligning images using feature matching.
"""

import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

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


def align(
    profile: pd.DataFrame,
    ref: str,
    method: str = "roma",
    max_rounds: int = 3,
    output_dir: Optional[str] = None,
    device: str = "cpu",
    resize: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Detect camera shifts and align images.

    This function detects camera position changes in a time series of images,
    segments the series into stable intervals, and aligns images to a
    reference image using feature matching.

    Supported matching methods:
        - OpenCV methods: "akaze", "sift"
        - IMM methods (requires imm package): "roma", "loftr", "sift-lightglue",
          "superpoint-lightglue", "minima-superpoint-lightglue", "tiny-roma",
          "minima-roma", "minima-loftr", "ufm", "rdd", "master"

    Memory-intensive methods (roma, loftr, etc.) are automatically resized to
    640px unless resize is explicitly specified. LightGlue-based methods can
    handle full-resolution images.

    Args:
        profile: DataFrame with 'filepath' and 'timestamp' columns.
        ref: Path to reference image.
        method: Matching method to use. Default is "roma".
        max_rounds: Maximum rounds for iterative alignment. Default is 3.
        output_dir: Directory to save aligned images.
        device: Device for IMM methods ("cpu" or "cuda"). Default is "cpu".
        resize: Resize images to this size for IMM methods.
            None means auto-resize for heavy methods.

    Returns:
        tuple: A tuple containing:
            - segments (pd.DataFrame): DataFrame with segment_id, start_idx, end_idx.
            - warp_params (pd.DataFrame): DataFrame with filepath and warp parameters,
              including match_status and num_matches.
            - aligned_dir (str): Path to directory containing aligned images.

    Example:
        >>> segments, warp_params, aligned_dir = align(profile, ref="ref.jpg")
        >>> segments.columns
        Index(['segment_id', 'start_idx', 'end_idx', 'anchor_filepath'])

        >>> # Using AKAZE (no external dependencies)
        >>> segments, warp_params, _ = align(profile, ref="ref.jpg", method="akaze")

        >>> # Using roma with GPU
        >>> segments, warp_params, _ = align(
        ...     profile, ref="ref.jpg", method="roma", device="cuda"
        ... )
    """
    # Create output directory
    if output_dir is None:
        output_dir = "aligned"
    aligned_dir = str(Path(output_dir).absolute())
    Path(aligned_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Detect shifts using phase correlation
    logger.info("Detecting camera shifts...")
    shifts = _detect_all_shifts(profile, ref)

    # Step 2: Segment time series
    logger.info("Segmenting time series...")
    segments_list = segment_timeseries(shifts, threshold=5.0)

    # Step 3: Select anchor images
    logger.info("Selecting anchor images...")
    anchors = select_anchor_images(segments_list, profile)

    # Step 4: Match anchors to reference and compute homographies
    logger.info(f"Matching to reference using {method} method...")
    warp_data = _match_all_to_reference(
        profile, ref, anchors, method, device=device, resize=resize
    )

    # Step 5: Apply warps and save aligned images
    logger.info("Applying warps...")
    _apply_and_save_warps(profile, warp_data, aligned_dir)

    # Build result DataFrames
    segments = pd.DataFrame(segments_list)
    warp_params = pd.DataFrame(warp_data)

    logger.info(
        f"Alignment complete: {len(segments)} segments, "
        f"{len(warp_params)} images aligned"
    )

    return segments, warp_params, aligned_dir


def phase_correlation_change_detect(
    img1: np.ndarray, img2: np.ndarray
) -> tuple[float, float]:
    """
    Detect shift between two images using phase correlation.

    Uses FFT-based phase correlation to estimate the translation
    between two images.

    Args:
        img1: First grayscale image.
        img2: Second grayscale image (same size as img1).

    Returns:
        tuple: (dx, dy) shift in pixels from img1 to img2.
    """
    # Ensure images are float32
    img1_float = np.float32(img1)
    img2_float = np.float32(img2)

    # Compute phase correlation
    shift, _ = cv2.phaseCorrelate(img1_float, img2_float)

    return (shift[0], shift[1])


def segment_timeseries(
    shifts: np.ndarray, threshold: float = 5.0
) -> list[dict]:
    """
    Segment time series into stable intervals based on cumulative shifts.

    Detects change points where the cumulative shift exceeds the threshold
    and creates segments representing stable camera positions.

    Args:
        shifts: Array of shape (N, 2) with (dx, dy) shifts for each frame.
        threshold: Threshold for detecting a significant shift change.

    Returns:
        list[dict]: List of segment dictionaries with keys:
            - segment_id: Integer segment identifier.
            - start_idx: Start index in the time series.
            - end_idx: End index in the time series.
    """
    n = len(shifts)
    if n == 0:
        return []

    segments = []
    current_start = 0
    segment_id = 0

    # Calculate cumulative shifts
    cumulative = np.cumsum(shifts, axis=0)

    for i in range(1, n):
        # Check if shift from segment start exceeds threshold
        shift_magnitude = np.sqrt(
            (cumulative[i, 0] - cumulative[current_start, 0]) ** 2
            + (cumulative[i, 1] - cumulative[current_start, 1]) ** 2
        )

        if shift_magnitude > threshold:
            # End current segment
            segments.append(
                {
                    "segment_id": segment_id,
                    "start_idx": current_start,
                    "end_idx": i - 1,
                }
            )
            segment_id += 1
            current_start = i

    # Add final segment
    segments.append(
        {
            "segment_id": segment_id,
            "start_idx": current_start,
            "end_idx": n - 1,
        }
    )

    return segments


def select_anchor_images(
    segments: list[dict], profile: pd.DataFrame
) -> list[dict]:
    """
    Select the best anchor image for each segment.

    Chooses the image with the best brightness (closest to 0.6) within
    each segment as the anchor for alignment.

    Args:
        segments: List of segment dictionaries.
        profile: DataFrame with 'filepath' and optionally 'brightness' columns.

    Returns:
        list[dict]: List of anchor dictionaries with keys:
            - segment_id: Segment identifier.
            - filepath: Path to anchor image.
            - idx: Index in profile.
    """
    anchors = []

    for seg in segments:
        start_idx = seg["start_idx"]
        end_idx = seg["end_idx"]

        # Get images in this segment
        segment_profile = profile.iloc[start_idx : end_idx + 1]

        if "brightness" in segment_profile.columns:
            # Select image with brightness closest to 0.6 (optimal)
            target_brightness = 0.6
            brightness_diff = (segment_profile["brightness"] - target_brightness).abs()
            best_idx = brightness_diff.idxmin()
        else:
            # Default to middle image
            best_idx = segment_profile.index[len(segment_profile) // 2]

        anchors.append(
            {
                "segment_id": seg["segment_id"],
                "filepath": profile.loc[best_idx, "filepath"],
                "idx": best_idx,
            }
        )

    return anchors


def estimate_homography(
    src_pts: np.ndarray, dst_pts: np.ndarray
) -> Optional[np.ndarray]:
    """
    Estimate homography matrix from matched point pairs.

    Uses RANSAC to robustly estimate the homography transformation.

    Args:
        src_pts: Source points array of shape (N, 2).
        dst_pts: Destination points array of shape (N, 2).

    Returns:
        np.ndarray or None: 3x3 homography matrix, or None if insufficient points.
    """
    if len(src_pts) < 4 or len(dst_pts) < 4:
        return None

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H


def apply_warp(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography transformation to an image.

    Args:
        img: Input image.
        H: 3x3 homography matrix.

    Returns:
        np.ndarray: Warped image.
    """
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, H, (w, h))
    return warped


def _detect_all_shifts(profile: pd.DataFrame, ref: str) -> np.ndarray:
    """
    Detect shifts for all images relative to adjacent frames.

    Args:
        profile: DataFrame with 'filepath' column.
        ref: Reference image path.

    Returns:
        np.ndarray: Array of shape (N, 2) with shifts.
    """
    n = len(profile)
    shifts = np.zeros((n, 2))

    # Load reference image
    ref_img = cv2.imread(ref, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        logger.warning(f"Could not read reference image: {ref}")
        return shifts

    prev_gray = ref_img

    for i, (_, row) in enumerate(profile.iterrows()):
        filepath = row["filepath"]
        curr_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        if curr_gray is None:
            logger.warning(f"Could not read image: {filepath}")
            continue

        # Resize if needed
        if curr_gray.shape != prev_gray.shape:
            curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))

        try:
            dx, dy = phase_correlation_change_detect(prev_gray, curr_gray)
            shifts[i] = [dx, dy]
        except Exception as e:
            logger.warning(f"Phase correlation failed for {filepath}: {e}")

        prev_gray = curr_gray

    return shifts


def _opencv_match(
    img1: np.ndarray,
    img2: np.ndarray,
    detector_type: str = "akaze",
    ratio: float = _DEFAULT_LOWE_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """
    OpenCV feature matching using AKAZE or SIFT.

    Args:
        img1: First image (BGR).
        img2: Second image (BGR).
        detector_type: Feature detector type: "akaze" or "sift".
        ratio: Lowe's ratio test threshold.

    Returns:
        tuple: (pts1, pts2) matched point pairs, shape (N, 2).
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

    # Lowe's ratio test
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
    """
    Feature matching using imm (image-matching-models) package.

    Args:
        path1: Path to first image.
        path2: Path to second image.
        method: IMM matching method name (e.g., "roma", "loftr", "sift-lightglue").
        device: Device for computation ("cpu" or "cuda").
        resize: Resize images to this size (optional).
        **kwargs: Additional arguments for imm's get_matcher.

    Returns:
        tuple: (pts1, pts2) matched point pairs, shape (N, 2).

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

    # Get original image sizes for coordinate scaling
    img1_cv = cv2.imread(path1)
    img2_cv = cv2.imread(path2)

    if img1_cv is None:
        raise ValueError(f"Could not read image: {path1}")
    if img2_cv is None:
        raise ValueError(f"Could not read image: {path2}")

    h1_org, w1_org = img1_cv.shape[:2]
    h2_org, w2_org = img2_cv.shape[:2]

    # Suppress torchvision deprecation warnings about 'pretrained' parameter
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        matcher = get_matcher(method, device=device, **kwargs)

    # Use matcher's load_image for proper preprocessing
    img0 = matcher.load_image(path1, resize=resize)
    img1 = matcher.load_image(path2, resize=resize)

    # Get the size after loading (to compute scale factors)
    # img shape is (C, H, W) or (1, C, H, W)
    if img0.dim() == 4:
        _, _, h0_loaded, w0_loaded = img0.shape
        _, _, h1_loaded, w1_loaded = img1.shape
    else:
        _, h0_loaded, w0_loaded = img0.shape
        _, h1_loaded, w1_loaded = img1.shape

    result = matcher(img0, img1)
    pts1 = result["matched_kpts0"]
    pts2 = result["matched_kpts1"]

    # Handle both torch tensors and numpy arrays
    if hasattr(pts1, "cpu"):
        pts1 = pts1.cpu().numpy()
        pts2 = pts2.cpu().numpy()

    if len(pts1) == 0:
        return (
            np.array([]).reshape(0, 2).astype("int32"),
            np.array([]).reshape(0, 2).astype("int32"),
        )

    # Scale keypoints back to original image coordinates
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
    """
    Match two images using specified method.

    Dispatches to appropriate matching function based on method name.

    Args:
        path1: Path to first image (anchor).
        path2: Path to second image (reference).
        method: Matching method ("akaze", "sift", or IMM methods like "roma", "loftr").
        device: Device for IMM methods ("cpu" or "cuda").
        resize: Resize for IMM methods.

    Returns:
        tuple: (pts1, pts2) matched point pairs in original coordinates, shape (N, 2).
    """
    method_lower = method.lower()

    if method_lower in ("akaze", "sift"):
        # OpenCV-based matching
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
        # IMM-based matching
        effective_resize = resize

        # Auto-resize for memory-intensive methods (non-LightGlue)
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


def _match_all_to_reference(
    profile: pd.DataFrame,
    ref: str,
    anchors: list[dict],
    method: str,
    device: str = "cpu",
    resize: Optional[int] = None,
) -> list[dict]:
    """
    Match all images to reference using feature matching.

    For each segment, matches the anchor image to the reference image to compute
    a homography transformation. This transformation is then applied to all images
    within that segment.

    Args:
        profile: DataFrame with 'filepath' column.
        ref: Reference image path.
        anchors: List of anchor dictionaries with keys:
            - segment_id: Segment identifier.
            - filepath: Path to anchor image.
            - idx: Index in profile.
        method: Matching method name ("akaze", "sift", or IMM methods).
        device: Device for IMM methods ("cpu" or "cuda").
        resize: Resize for IMM methods (None for auto).

    Returns:
        list[dict]: List of warp parameter dictionaries with keys:
            - filepath: Image path
            - H_00 through H_22: Homography matrix elements
            - match_status: "success" or "failed"
            - num_matches: Number of matched points (for anchor images)
    """
    warp_data = []

    # Build a mapping from segment_id to anchor info and homography
    segment_homographies = {}

    # First, compute homography for each anchor to reference
    for anchor in anchors:
        segment_id = anchor["segment_id"]
        anchor_path = anchor["filepath"]

        logger.debug(f"Matching anchor {anchor_path} to reference {ref}")

        try:
            pts_anchor, pts_ref = _match_images(
                anchor_path, ref, method=method, device=device, resize=resize
            )

            if len(pts_anchor) >= 4:
                H = estimate_homography(
                    pts_anchor.astype(np.float32), pts_ref.astype(np.float32)
                )
                if H is not None:
                    segment_homographies[segment_id] = {
                        "H": H,
                        "num_matches": len(pts_anchor),
                        "status": "success",
                    }
                    logger.debug(
                        f"Segment {segment_id}: matched {len(pts_anchor)} points"
                    )
                else:
                    segment_homographies[segment_id] = {
                        "H": np.eye(3, dtype=np.float32),
                        "num_matches": len(pts_anchor),
                        "status": "failed",
                    }
                    logger.warning(
                        f"Segment {segment_id}: homography estimation failed"
                    )
            else:
                segment_homographies[segment_id] = {
                    "H": np.eye(3, dtype=np.float32),
                    "num_matches": len(pts_anchor),
                    "status": "failed",
                }
                logger.warning(
                    f"Segment {segment_id}: insufficient matches ({len(pts_anchor)})"
                )
        except Exception as e:
            logger.warning(f"Segment {segment_id}: matching failed: {e}")
            segment_homographies[segment_id] = {
                "H": np.eye(3, dtype=np.float32),
                "num_matches": 0,
                "status": "failed",
            }

    # Build segment index lookup for each image
    # Determine which segment each image belongs to
    anchor_by_idx = {a["idx"]: a["segment_id"] for a in anchors}

    # Build segment ranges from anchors
    # Sort anchors by index
    sorted_anchors = sorted(anchors, key=lambda x: x["idx"])
    segment_ranges = []

    for i, anchor in enumerate(sorted_anchors):
        start_idx = anchor["idx"]
        if i + 1 < len(sorted_anchors):
            end_idx = sorted_anchors[i + 1]["idx"] - 1
        else:
            end_idx = len(profile) - 1
        segment_ranges.append(
            {"segment_id": anchor["segment_id"], "start": start_idx, "end": end_idx}
        )

    # Handle images before first anchor
    if sorted_anchors and sorted_anchors[0]["idx"] > 0:
        segment_ranges.insert(
            0,
            {
                "segment_id": sorted_anchors[0]["segment_id"],
                "start": 0,
                "end": sorted_anchors[0]["idx"] - 1,
            },
        )

    # Assign segment_id to each profile row
    def get_segment_id(idx: int) -> int:
        for sr in segment_ranges:
            if sr["start"] <= idx <= sr["end"]:
                return sr["segment_id"]
        # Default to first segment
        return sorted_anchors[0]["segment_id"] if sorted_anchors else 0

    # Apply homography to all images in each segment
    for idx, row in profile.iterrows():
        filepath = row["filepath"]
        segment_id = get_segment_id(idx)

        if segment_id in segment_homographies:
            H = segment_homographies[segment_id]["H"]
            status = segment_homographies[segment_id]["status"]
            num_matches = segment_homographies[segment_id]["num_matches"]
        else:
            H = np.eye(3, dtype=np.float32)
            status = "failed"
            num_matches = 0

        warp_data.append(
            {
                "filepath": filepath,
                "H_00": float(H[0, 0]),
                "H_01": float(H[0, 1]),
                "H_02": float(H[0, 2]),
                "H_10": float(H[1, 0]),
                "H_11": float(H[1, 1]),
                "H_12": float(H[1, 2]),
                "H_20": float(H[2, 0]),
                "H_21": float(H[2, 1]),
                "H_22": float(H[2, 2]),
                "segment_id": segment_id,
                "match_status": status,
                "num_matches": num_matches if idx in anchor_by_idx else None,
            }
        )

    return warp_data


def _apply_and_save_warps(
    profile: pd.DataFrame, warp_data: list[dict], output_dir: str
) -> None:
    """
    Apply warps to images and save to output directory.

    Args:
        profile: DataFrame with 'filepath' column.
        warp_data: List of warp parameter dictionaries.
        output_dir: Directory to save aligned images.
    """
    for warp_entry in warp_data:
        filepath = warp_entry["filepath"]

        # Reconstruct homography matrix
        H = np.array(
            [
                [warp_entry["H_00"], warp_entry["H_01"], warp_entry["H_02"]],
                [warp_entry["H_10"], warp_entry["H_11"], warp_entry["H_12"]],
                [warp_entry["H_20"], warp_entry["H_21"], warp_entry["H_22"]],
            ],
            dtype=np.float32,
        )

        # Read image
        img = cv2.imread(filepath)
        if img is None:
            logger.warning(f"Could not read image: {filepath}")
            continue

        # Apply warp
        aligned = apply_warp(img, H)

        # Save aligned image
        output_path = Path(output_dir) / Path(filepath).name
        cv2.imwrite(str(output_path), aligned)
