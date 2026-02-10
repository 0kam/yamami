"""
Semantic segmentation module for yamami.

Provides image segmentation using Meta SAM 3 with text prompts
(e.g., sky, cloud, fog, snow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from yamami.logging import get_logger

logger = get_logger(__name__)

# Default prompts for semantic segmentation (order = priority, first is highest)
DEFAULT_PROMPTS = ["fog", "cloud", "sky", "sun", "reflection"]

# Default color map (BGR format) - will be auto-generated if not specified
DEFAULT_COLORS = {
    "fog": (200, 200, 200),       # Gray
    "cloud": (255, 255, 255),     # White
    "sky": (255, 180, 100),       # Light blue
    "sun": (0, 255, 255),         # Yellow
    "reflection": (0, 255, 0),    # Green
    "snow": (255, 200, 200),      # Light cyan
    "raindrop": (255, 0, 0),      # Blue
}

# Global model cache to avoid reloading
_model_cache: dict = {}


def _generate_colors(prompts: list[str], user_colors: Optional[dict] = None) -> dict:
    """
    Generate colors for prompts.

    Uses user-specified colors first, then defaults, then auto-generates.

    Args:
        prompts: List of prompt names.
        user_colors: Optional user-specified colors (BGR format).

    Returns:
        dict: Mapping of prompt -> BGR color tuple.
    """
    import colorsys

    colors = {}
    used_colors = set()

    # First pass: use user-specified or default colors
    for prompt in prompts:
        if user_colors and prompt in user_colors:
            colors[prompt] = user_colors[prompt]
            used_colors.add(user_colors[prompt])
        elif prompt in DEFAULT_COLORS:
            colors[prompt] = DEFAULT_COLORS[prompt]
            used_colors.add(DEFAULT_COLORS[prompt])

    # Second pass: auto-generate for remaining prompts
    remaining = [p for p in prompts if p not in colors]
    if remaining:
        # Generate distinct colors using HSV
        n = len(remaining)
        for i, prompt in enumerate(remaining):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors[prompt] = bgr

    return colors


def _get_device(device: str) -> str:
    """
    Resolve device string to actual device.

    Args:
        device: Device specification ("auto", "cpu", or "cuda").

    Returns:
        str: Resolved device string ("cpu" or "cuda").
    """
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def _load_models(device: str) -> tuple:
    """
    Load or retrieve cached SAM3 model and processor.

    Args:
        device: "cuda" or "cpu". SAM3 requires CUDA.

    Returns:
        tuple: (model, processor)
    """
    cache_key = f"sam3:{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if device != "cuda":
        raise RuntimeError(
            "SAM3 requires a CUDA-capable GPU (device='cuda')."
        )

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        raise ImportError(
            "Semantic segmentation requires 'sam3'. "
            "Install from https://github.com/facebookresearch/sam3"
        ) from e

    logger.info("Loading SAM3 image model")
    model = build_sam3_image_model()
    model.to(device)
    model.eval()
    processor = Sam3Processor(model)

    _model_cache[cache_key] = (model, processor)
    return model, processor


def _resize_for_inference(img_rgb: np.ndarray, resize: int) -> tuple[np.ndarray, float]:
    if resize <= 0:
        return img_rgb, 1.0

    h, w = img_rgb.shape[:2]
    max_dim = max(h, w)
    if max_dim <= resize:
        return img_rgb, 1.0

    scale = resize / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _extract_masks(output: dict) -> Optional[np.ndarray]:
    masks = output.get("masks")
    if masks is None:
        return None
    # Handle torch tensors or numpy arrays
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    return masks


def _combine_instance_masks(masks: np.ndarray) -> np.ndarray:
    """
    Combine multiple instance masks into a single binary mask.

    Handles masks of shape:
    - (H, W): Single mask
    - (N, H, W): N instance masks
    - (N, 1, H, W): N instance masks with batch dim
    """
    if masks is None:
        return None

    # Squeeze all dimensions of size 1 except the last two (H, W)
    while masks.ndim > 2 and masks.shape[0] == 1:
        masks = masks[0]

    # Handle (N, 1, H, W) -> (N, H, W)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]

    if masks.ndim == 2:
        # Single mask (H, W)
        return masks
    elif masks.ndim == 3:
        # Multiple masks (N, H, W) -> combine with OR
        return np.any(masks, axis=0)
    else:
        # Fallback: try to reduce to 2D
        return np.any(masks, axis=tuple(range(masks.ndim - 2)))


def _segment_single_image(
    img: np.ndarray,
    prompts: list[str],
    model_bundle: tuple,
    resize: int,
    device: str,
) -> dict[str, np.ndarray]:
    """
    Segment a single image using SAM3 text prompts.

    Args:
        img: Input image (BGR).
        prompts: Text prompts.
        model_bundle: (model, processor)
        resize: Max side length for inference (0 = no resize).
        device: Device string.

    Returns:
        dict: prompt -> binary mask (bool array)
    """
    from PIL import Image

    _, processor = model_bundle

    original_h, original_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    infer_img, scale = _resize_for_inference(img_rgb, resize)
    pil_img = Image.fromarray(infer_img)

    masks_dict: dict[str, np.ndarray] = {}
    for prompt in prompts:
        # Create fresh inference state for each prompt to avoid state pollution
        inference_state = processor.set_image(pil_img)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks = _extract_masks(output)
        combined = _combine_instance_masks(masks)
        if combined is None:
            mask_bool = np.zeros(infer_img.shape[:2], dtype=bool)
        else:
            mask_bool = combined > 0

        if scale != 1.0:
            # Resize mask back to original size
            mask_uint8 = mask_bool.astype(np.uint8) * 255
            mask_uint8 = cv2.resize(
                mask_uint8,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_bool = mask_uint8 > 0
        masks_dict[prompt] = mask_bool

    # Apply priority: earlier prompts take precedence over later ones
    # e.g., prompts=["fog", "cloud", "sky"] -> fog > cloud > sky
    for i, prompt in enumerate(prompts):
        if prompt not in masks_dict:
            continue
        # Remove all higher-priority masks from this mask
        for higher_priority_prompt in prompts[:i]:
            if higher_priority_prompt in masks_dict:
                masks_dict[prompt] = masks_dict[prompt] & ~masks_dict[higher_priority_prompt]

    return masks_dict


def _calculate_ratio(mask: np.ndarray) -> float:
    """
    Calculate the ratio of masked pixels to total pixels.

    Args:
        mask: Binary mask array (bool).

    Returns:
        float: Ratio of masked area (0.0 to 1.0).
    """
    if mask.size == 0:
        return 0.0
    return float(np.sum(mask) / mask.size)


def segment(
    profile: pd.DataFrame,
    prompts: Optional[list[str]] = None,
    colors: Optional[dict] = None,
    resize: int = 512,
    device: str = "auto",
    output_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run semantic segmentation on images using SAM3.

    Args:
        profile: DataFrame with 'path' column containing image paths.
        prompts: List of text prompts for segmentation. Defaults to
            ["fog", "cloud", "sky", "sun", "reflection"].
        colors: Optional dict mapping prompt -> BGR color tuple for preview.
            Auto-generated if not specified.
        resize: Target size for image resizing before segmentation.
            Default is 512. Set to 0 for no resizing.
        device: Device for model inference ("auto", "cpu", or "cuda").
            Default is "auto".
        output_dir: Optional directory to save segmentation outputs.

    Returns:
        tuple: (labels_df, masks_dict)
            - labels_df: DataFrame with columns 'path', 'prompt', 'ratio'
              (long format, one row per path-prompt combination).
            - masks_dict: Dictionary mapping filepath to dict of prompt -> bool mask.
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS.copy()

    # Generate colors for preview
    color_map = _generate_colors(prompts, colors)

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    resolved_device = _get_device(device)
    logger.info(f"Using device: {resolved_device}")

    model_bundle = _load_models(resolved_device)

    labels_data = []
    masks_dict = {}

    for _, row in profile.iterrows():
        filepath = row["path"]

        img = cv2.imread(filepath)
        if img is None:
            logger.warning(f"Failed to read image: {filepath}")
            for prompt in prompts:
                labels_data.append({
                    "path": filepath,
                    "prompt": prompt,
                    "ratio": 0.0,
                })
            masks_dict[filepath] = {p: np.zeros((1, 1), dtype=bool) for p in prompts}
            continue

        mask_entry = _segment_single_image(
            img, prompts, model_bundle, resize, resolved_device
        )

        for prompt in prompts:
            ratio = _calculate_ratio(mask_entry[prompt])
            labels_data.append({
                "path": filepath,
                "prompt": prompt,
                "ratio": ratio,
            })

        masks_dict[filepath] = mask_entry

        if output_dir is not None:
            _save_masks(filepath, mask_entry, output_dir, color_map, original_img=img)

    labels = pd.DataFrame(labels_data)

    # Save labels CSV
    if output_dir is not None:
        labels_path = Path(output_dir) / "segment.csv"
        labels.to_csv(labels_path, index=False)

    logger.info(f"Segmented {len(profile)} images with {len(prompts)} prompts")

    return labels, masks_dict


def _save_masks(
    filepath: str,
    masks: dict[str, np.ndarray],
    output_dir: str,
    color_map: dict,
    original_img: Optional[np.ndarray] = None,
    preview_width: int = 2048,
) -> None:
    """
    Save segmentation masks to output directory.

    Saves binary masks as compressed .npz files and a side-by-side preview with legend.

    Args:
        filepath: Original image filepath.
        masks: Dictionary mapping prompt names to bool mask arrays.
        output_dir: Directory to save masks.
        color_map: Dict mapping prompt -> BGR color tuple.
        original_img: Original image for preview (BGR format).
        preview_width: Width of the preview image (default 2048).
    """
    basename = Path(filepath).stem
    output_path = Path(output_dir)

    # Save masks subdirectory
    masks_dir = output_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Save individual binary masks as compressed numpy arrays
    for prompt, mask in masks.items():
        mask_filename = f"{basename}_{prompt}.npz"
        mask_path = masks_dir / mask_filename
        np.savez_compressed(str(mask_path), mask=mask)

    # Save side-by-side preview with legend in preview subdirectory
    if masks and original_img is not None:
        h, w = original_img.shape[:2]

        # Create colored mask image
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for prompt, mask in masks.items():
            color = color_map.get(prompt, (0, 255, 0))
            colored[mask] = color

        # Concatenate horizontally
        combined = np.hstack([original_img, colored])

        # Add legend
        combined = _add_legend(combined, masks, color_map)

        # Resize to preview_width
        combined_h, combined_w = combined.shape[:2]
        if combined_w > preview_width:
            scale = preview_width / combined_w
            new_h = int(combined_h * scale)
            combined = cv2.resize(combined, (preview_width, new_h), interpolation=cv2.INTER_AREA)

        # Save preview in subdirectory
        preview_dir = output_path / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_filename = f"{basename}_preview.jpg"
        preview_path = preview_dir / preview_filename
        cv2.imwrite(str(preview_path), combined)


def _add_legend(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    color_map: dict,
    legend_height: int = 400,
    font_scale: float = 6.0,
) -> np.ndarray:
    """
    Add a legend bar to the bottom of the image.

    Args:
        image: Input image (BGR).
        masks: Dictionary of masks (used for ordering).
        color_map: Dict mapping prompt -> BGR color tuple.
        legend_height: Height of legend bar in pixels.
        font_scale: Font scale for text.

    Returns:
        Image with legend added at bottom.
    """
    h, w = image.shape[:2]

    # Create legend bar
    legend = np.zeros((legend_height, w, 3), dtype=np.uint8)
    legend[:] = (40, 40, 40)  # Dark gray background

    # Calculate positions for each label
    prompts = list(masks.keys())
    n_prompts = len(prompts)
    if n_prompts == 0:
        return image

    spacing = w // (n_prompts + 1)

    for i, prompt in enumerate(prompts):
        x = spacing * (i + 1)
        color = color_map.get(prompt, (0, 255, 0))

        # Draw color box
        box_size = 180
        box_x = x - 400
        box_y = (legend_height - box_size) // 2
        cv2.rectangle(legend, (box_x, box_y), (box_x + box_size, box_y + box_size), color, -1)
        cv2.rectangle(legend, (box_x, box_y), (box_x + box_size, box_y + box_size), (255, 255, 255), 4)

        # Draw text
        text_x = box_x + box_size + 30
        text_y = legend_height // 2 + 40
        cv2.putText(legend, prompt, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 8, cv2.LINE_AA)

    # Stack image and legend
    return np.vstack([image, legend])
