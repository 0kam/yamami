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

# Default prompts for semantic segmentation
DEFAULT_PROMPTS = ["sky", "cloud", "fog", "snow"]

# Global model cache to avoid reloading
_model_cache: dict = {}


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
    if masks is None:
        return None
    if masks.ndim == 2:
        combined = masks
    elif masks.ndim == 3:
        combined = np.any(masks, axis=0)
    else:
        return None
    return combined


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
        dict: prompt -> binary mask (0/255)
    """
    from PIL import Image

    _, processor = model_bundle

    original_h, original_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    infer_img, scale = _resize_for_inference(img_rgb, resize)
    pil_img = Image.fromarray(infer_img)

    inference_state = processor.set_image(pil_img)

    masks_dict: dict[str, np.ndarray] = {}
    for prompt in prompts:
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks = _extract_masks(output)
        combined = _combine_instance_masks(masks)
        if combined is None:
            mask_uint8 = np.zeros(infer_img.shape[:2], dtype=np.uint8)
        else:
            mask_uint8 = (combined > 0).astype(np.uint8) * 255

        if scale != 1.0:
            mask_uint8 = cv2.resize(
                mask_uint8,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )
        masks_dict[prompt] = mask_uint8

    return masks_dict


def _calculate_ratio(mask: np.ndarray) -> float:
    """
    Calculate the ratio of masked pixels to total pixels.

    Args:
        mask: Binary mask array (0 or 255).

    Returns:
        float: Ratio of masked area (0.0 to 1.0).
    """
    if mask.size == 0:
        return 0.0
    return float(np.sum(mask > 0) / mask.size)


def segment(
    profile: pd.DataFrame,
    prompts: Optional[list[str]] = None,
    resize: int = 512,
    device: str = "auto",
    output_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run semantic segmentation on images using SAM3.

    Args:
        profile: DataFrame with 'filepath' column containing image paths.
        prompts: List of text prompts for segmentation. Defaults to
            ["sky", "cloud", "fog", "snow"].
        resize: Target size for image resizing before segmentation.
            Default is 512.
        device: Device for model inference ("auto", "cpu", or "cuda").
            Default is "auto".
        output_dir: Optional directory to save segmentation outputs.

    Returns:
        tuple: (labels, masks)
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS.copy()

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    resolved_device = _get_device(device)
    logger.info(f"Using device: {resolved_device}")

    model_bundle = _load_models(resolved_device)

    labels_data = []
    masks_dict = {}

    for _, row in profile.iterrows():
        filepath = row["filepath"]

        img = cv2.imread(filepath)
        if img is None:
            logger.warning(f"Failed to read image: {filepath}")
            label_entry = {"filepath": filepath, "brightness": 0.0}
            for prompt in prompts:
                label_entry[f"{prompt}_ratio"] = 0.0
            labels_data.append(label_entry)
            masks_dict[filepath] = {p: np.zeros((1, 1), dtype=np.uint8) for p in prompts}
            continue

        brightness = _calculate_brightness(img)

        mask_entry = _segment_single_image(
            img, prompts, model_bundle, resize, resolved_device
        )
        label_entry = {"filepath": filepath, "brightness": brightness}
        for prompt in prompts:
            ratio = _calculate_ratio(mask_entry[prompt])
            label_entry[f"{prompt}_ratio"] = ratio

        labels_data.append(label_entry)
        masks_dict[filepath] = mask_entry

        if output_dir is not None:
            _save_masks(filepath, mask_entry, output_dir)

    labels = pd.DataFrame(labels_data)

    logger.info(f"Segmented {len(profile)} images with {len(prompts)} prompts")

    return labels, masks_dict


def _calculate_brightness(img: np.ndarray) -> float:
    """
    Calculate normalized brightness of an image.

    Converts the image to grayscale and computes the mean pixel value
    normalized to the range [0, 1].

    Args:
        img: Input image as numpy array (BGR format).

    Returns:
        float: Normalized brightness value between 0 and 1.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    mean_value = np.mean(gray)
    return float(mean_value / 255.0)


def _save_masks(
    filepath: str, masks: dict[str, np.ndarray], output_dir: str
) -> None:
    """
    Save segmentation masks to output directory.

    Args:
        filepath: Original image filepath.
        masks: Dictionary mapping prompt names to mask arrays.
        output_dir: Directory to save masks.
    """
    basename = Path(filepath).stem

    for prompt, mask in masks.items():
        mask_filename = f"{basename}_{prompt}_mask.png"
        mask_path = Path(output_dir) / mask_filename
        cv2.imwrite(str(mask_path), mask)
