"""
Semantic segmentation module for yamami.

Provides image segmentation using Meta SAM 3 with text prompts
(e.g., sky, cloud, fog, snow).

Performance optimizations:
- Strategy A: Instance prompts share one image encoder pass per image
- Strategy B: Score-map prompts share tiled image encoder passes
- Strategy C: PyTorch DataLoader prefetches images from disk in worker processes
- Strategy D: Mask saving runs in background threads via ThreadPoolExecutor
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from yamami.logging import get_logger

logger = get_logger(__name__)

# Default prompts for semantic segmentation (order = priority, first is highest)
DEFAULT_PROMPTS = [
    "fog",
    "cloud",
    "sky",
    "sun",
    "reflection",
    "snow",
    "mountain",
]
DEFAULT_MODE = {
    "default": "fused",
    "fog": "instance",
    "cloud": "semantic_only",
}

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

SUPPORTED_MODES = {
    "instance",
    "instance_only",
    "semantic",
    "semantic_only",
    "fused",
}


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


def _resolve_checkpoint_path(checkpoint_path: Optional[str]) -> Optional[str]:
    """
    Resolve the SAM3 checkpoint path from argument or SAM3_CHECKPOINT_PATH env var.

    Returns None when neither is set (caller will fall back to HF download).
    Raises FileNotFoundError if a path is provided but does not exist.
    """
    resolved = checkpoint_path or os.environ.get("SAM3_CHECKPOINT_PATH")
    if resolved is None:
        return None
    if not Path(resolved).is_file():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {resolved}. "
            "Set SAM3_CHECKPOINT_PATH or pass checkpoint_path to a valid sam3.pt file."
        )
    return resolved


def _load_models(device: str, checkpoint_path: Optional[str] = None) -> tuple:
    """
    Load or retrieve cached SAM3 model and processor.

    Args:
        device: "cuda" or "cpu". SAM3 requires CUDA.
        checkpoint_path: Optional path to a local sam3.pt checkpoint. When
            omitted, falls back to the SAM3_CHECKPOINT_PATH environment
            variable, and finally to HuggingFace download (gated by Meta).

    Returns:
        tuple: (model, processor)
    """
    resolved_ckpt = _resolve_checkpoint_path(checkpoint_path)
    cache_key = f"sam3:{device}:{resolved_ckpt or 'hf'}"
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

    if resolved_ckpt is not None:
        logger.info(f"Loading SAM3 image model from local checkpoint: {resolved_ckpt}")
        model = build_sam3_image_model(
            checkpoint_path=resolved_ckpt,
            load_from_HF=False,
        )
    else:
        logger.info("Loading SAM3 image model (HuggingFace download)")
        model = build_sam3_image_model()
    model.to(device)
    model.eval()
    processor = Sam3Processor(model)

    _model_cache[cache_key] = (model, processor)
    return model, processor


def _normalise_modes(
    prompts: list[str],
    mode: str | dict[str, str],
) -> dict[str, str]:
    """Resolve per-prompt SAM3 output mode."""
    if isinstance(mode, str):
        mapping = {prompt: mode for prompt in prompts}
    else:
        default_mode = mode.get("default", "fused")
        mapping = {
            prompt: mode.get(prompt, default_mode)
            for prompt in prompts
        }

    aliases = {
        "semantic": "semantic_only",
    }
    mapping = {
        prompt: aliases.get(str(mode).lower(), str(mode).lower())
        for prompt, mode in mapping.items()
    }
    unsupported = sorted(set(mapping.values()) - SUPPORTED_MODES)
    if unsupported:
        raise ValueError(
            f"Unsupported SAM3 prompt mode(s): {unsupported}. "
            f"Supported modes: {sorted(SUPPORTED_MODES)}"
        )
    return mapping


def _normalise_prompt_floats(
    prompts: list[str],
    value: float | dict[str, float],
    default: float,
) -> dict[str, float]:
    if isinstance(value, dict):
        default_value = float(value.get("default", default))
        return {
            prompt: float(value.get(prompt, default_value))
            for prompt in prompts
        }
    return {prompt: float(value) for prompt in prompts}


def _combine_score_branches(
    inst_score,
    semantic_score,
    score_mode: str,
):
    if score_mode == "instance_only":
        return inst_score
    if score_mode == "semantic_only":
        return semantic_score
    return inst_score.maximum(semantic_score)


def _tile_origins(length: int, crop: int, stride: int) -> list[int]:
    if crop <= 0 or length <= crop:
        return [0]
    origins = list(range(0, length - crop + 1, stride))
    last = length - crop
    if origins[-1] != last:
        origins.append(last)
    return origins


def _sam3_score_batch(
    pil_imgs: list,
    prompts: list[str],
    model_bundle: tuple,
    device: str,
    mode_map: dict[str, str],
    confidence_map: dict[str, float],
) -> dict[str, list[np.ndarray]]:
    """Return per-pixel SAM3 score maps for image batch x prompts."""
    import torch
    import torch.nn.functional as F
    from sam3.model.data_misc import FindStage

    _, processor = model_bundle
    sizes = [img.size for img in pil_imgs]
    image_count = len(pil_imgs)
    prompt_count = len(prompts)
    combo_count = image_count * prompt_count

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image_batch(pil_imgs)
        text_outputs = processor.model.backbone.forward_text(
            prompts, device=device,
        )
        state["backbone_out"].update(text_outputs)
        geometric_prompt = processor.model._get_dummy_prompt(
            num_prompts=combo_count,
        )
        find_stage = FindStage(
            img_ids=torch.arange(
                image_count, device=device, dtype=torch.long,
            ).repeat_interleave(prompt_count),
            text_ids=torch.arange(
                prompt_count, device=device, dtype=torch.long,
            ).repeat(image_count),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        outputs = processor.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

        logits = outputs["pred_logits"].sigmoid().squeeze(-1).float()
        presence = outputs["presence_logit_dec"].sigmoid().float().flatten()
        logits = logits * presence.view(-1, 1)

        inst_masks = outputs.get("pred_masks")
        semantic_masks = outputs.get("semantic_seg")

        scores: dict[str, list[np.ndarray]] = {
            prompt: [] for prompt in prompts
        }
        for image_idx, (width, height) in enumerate(sizes):
            for prompt_idx, prompt in enumerate(prompts):
                out_idx = image_idx * prompt_count + prompt_idx
                mode = mode_map[prompt]
                confidence_threshold = confidence_map[prompt]

                inst_score = torch.zeros(
                    (height, width), device=device, dtype=torch.float32,
                )
                semantic_score = torch.zeros(
                    (height, width), device=device, dtype=torch.float32,
                )

                if inst_masks is not None and inst_masks.shape[1] > 0:
                    keep = logits[out_idx] > confidence_threshold
                    if bool(keep.any()):
                        inst = inst_masks[out_idx, keep].float()
                        if inst.shape[-2:] != (height, width):
                            inst = F.interpolate(
                                inst.unsqueeze(1),
                                size=(height, width),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(1)
                        inst = inst.sigmoid()
                        obj_scores = logits[out_idx, keep].view(-1, 1, 1)
                        inst_score = (inst * obj_scores).max(dim=0).values

                if semantic_masks is not None:
                    semantic = semantic_masks[out_idx].float()
                    if semantic.shape[-2:] != (height, width):
                        semantic = F.interpolate(
                            semantic.unsqueeze(0),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    semantic_score = semantic.sigmoid().squeeze()

                fused = _combine_score_branches(
                    inst_score, semantic_score, mode,
                )
                fused = fused * presence[out_idx]
                scores[prompt].append(fused.cpu().numpy().astype(np.float32))

    return scores


def _sam3_tiled_scores(
    image,
    prompts: list[str],
    model_bundle: tuple,
    device: str,
    tile_size: int,
    tile_stride: int,
    batch_size: int,
    mode_map: dict[str, str],
    confidence_map: dict[str, float],
) -> dict[str, np.ndarray]:
    """Score prompts on a full image, sharing tile image encodes."""
    width, height = image.size
    if tile_size <= 0:
        batched = _sam3_score_batch(
            [image], prompts, model_bundle, device, mode_map, confidence_map,
        )
        return {prompt: scores[0] for prompt, scores in batched.items()}

    xs = _tile_origins(width, tile_size, tile_stride)
    ys = _tile_origins(height, tile_size, tile_stride)
    acc = {
        prompt: np.zeros((height, width), dtype=np.float32)
        for prompt in prompts
    }
    cnt = np.zeros((height, width), dtype=np.float32)

    tiles = [
        (x, y, image.crop((
            x, y, min(x + tile_size, width), min(y + tile_size, height),
        )))
        for y in ys
        for x in xs
    ]

    tile_batch_size = max(1, int(batch_size) // max(1, len(prompts)))
    for start in range(0, len(tiles), tile_batch_size):
        batch = tiles[start:start + tile_batch_size]
        scores_by_prompt = _sam3_score_batch(
            [patch for _, _, patch in batch],
            prompts,
            model_bundle,
            device,
            mode_map,
            confidence_map,
        )
        batch_weight = np.zeros((height, width), dtype=np.float32)
        for tile_idx, (x, y, _) in enumerate(batch):
            first_prompt = prompts[0]
            ph, pw = scores_by_prompt[first_prompt][tile_idx].shape
            wy = np.hanning(ph) if ph > 1 else np.ones(ph)
            wx = np.hanning(pw) if pw > 1 else np.ones(pw)
            win = np.outer(
                np.maximum(wy, 0.08),
                np.maximum(wx, 0.08),
            ).astype(np.float32)
            for prompt in prompts:
                score = scores_by_prompt[prompt][tile_idx]
                acc[prompt][y:y + ph, x:x + pw] += score * win
            batch_weight[y:y + ph, x:x + pw] += win
        cnt += batch_weight

    return {
        prompt: prompt_acc / np.maximum(cnt, 1e-6)
        for prompt, prompt_acc in acc.items()
    }


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


# ---------------------------------------------------------------------------
# Strategy B: PyTorch DataLoader for I/O prefetching
# ---------------------------------------------------------------------------

class _ImageDataset(Dataset):
    """PyTorch Dataset that prefetches image reading and resizing in workers."""

    def __init__(self, filepaths: list[str], resize: int):
        self.filepaths = filepaths
        self.resize = resize

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> tuple:
        filepath = self.filepaths[idx]
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            return filepath, None, None, 1.0
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb, scale = _resize_for_inference(img_rgb, self.resize)
        return filepath, img_bgr, resized_rgb, scale


def _collate_passthrough(batch: list) -> tuple:
    """Pass through a single item without default tensor conversion."""
    return batch[0]


# ---------------------------------------------------------------------------
# Strategy A: Image encoder runs once; deepcopy state for each prompt
# ---------------------------------------------------------------------------

def _segment_single_image(
    resized_rgb: np.ndarray,
    original_hw: tuple[int, int],
    prompts: list[str],
    model_bundle: tuple,
    scale: float,
) -> dict[str, np.ndarray]:
    """
    Segment a single image using SAM3 text prompts.

    Encodes the image once via set_image(), then deepcopies the state
    for each prompt to avoid re-running the expensive backbone encoder.

    Args:
        resized_rgb: Pre-resized RGB image for inference.
        original_hw: Original image (height, width) for mask upscaling.
        prompts: Text prompts.
        model_bundle: (model, processor)
        scale: Resize scale factor (1.0 = no resize).

    Returns:
        dict: prompt -> binary mask (bool array at original resolution)
    """
    from PIL import Image

    _, processor = model_bundle

    pil_img = Image.fromarray(resized_rgb)

    # Strategy A: encode image once (expensive backbone forward pass)
    base_state = processor.set_image(pil_img)

    original_h, original_w = original_hw
    masks_dict: dict[str, np.ndarray] = {}

    for prompt in prompts:
        # SAM3's set_text_prompt mutates the state, so deepcopy before each call.
        # deepcopy copies GPU tensors via CUDA memcpy which is far cheaper than
        # re-running the backbone encoder.
        state = copy.deepcopy(base_state)
        output = processor.set_text_prompt(state=state, prompt=prompt)
        masks = _extract_masks(output)
        combined = _combine_instance_masks(masks)
        if combined is None:
            mask_bool = np.zeros(resized_rgb.shape[:2], dtype=bool)
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


def _apply_prompt_priority(
    masks: dict[str, np.ndarray],
    prompts: list[str],
    image_hw: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Apply prompt order as priority, returning every requested prompt."""
    h, w = image_hw
    prioritized: dict[str, np.ndarray] = {}
    occupied = np.zeros((h, w), dtype=bool)
    for prompt in prompts:
        mask = masks.get(prompt)
        if mask is None:
            mask = np.zeros((h, w), dtype=bool)
        else:
            mask = mask.astype(bool)
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
        visible = mask & ~occupied
        prioritized[prompt] = visible
        occupied |= visible
    return prioritized


def segment(
    profile: pd.DataFrame,
    prompts: Optional[list[str]] = None,
    colors: Optional[dict] = None,
    resize: int = 512,
    device: str = "auto",
    output_dir: Optional[str] = None,
    num_workers: int = 2,
    checkpoint_path: Optional[str] = None,
    mode: str | dict[str, str] | None = None,
    tile_size: Optional[int] = None,
    tile_stride: Optional[int] = None,
    batch_size: int = 4,
    threshold: float | dict[str, float] = 0.5,
    confidence_threshold: float | dict[str, float] = 0.5,
    save_scores: bool = False,
    save_score_preview: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Run semantic segmentation on images using SAM3.

    Args:
        profile: DataFrame with 'path' column containing image paths.
        prompts: List of text prompts for segmentation. Defaults to
            ["fog", "cloud", "sky", "sun", "reflection", "snow", "mountain"].
        colors: Optional dict mapping prompt -> BGR color tuple for preview.
            Auto-generated if not specified.
        resize: Target size for image resizing before segmentation.
            Default is 512. Set to 0 for no resizing.
        device: Device for model inference ("auto", "cpu", or "cuda").
            Default is "auto".
        output_dir: Optional directory to save segmentation outputs.
        num_workers: Number of DataLoader workers for I/O prefetching.
            Default is 2. Set to 0 to disable prefetching.
        checkpoint_path: Optional path to a local SAM3 checkpoint (sam3.pt).
            Falls back to the SAM3_CHECKPOINT_PATH env var, then to the
            HuggingFace gated download.
        mode: SAM3 output mode or per-prompt mapping. Supported modes are
            ``"fused"``, ``"semantic_only"``, ``"instance_only"``, and
            ``"instance"``. Defaults to ``"fused"`` for all prompts.
        tile_size: Optional global tile size for score-map modes. Set to
            ``None`` or ``0`` to score the whole image at once.
        tile_stride: Optional global tile stride for score-map modes. When
            omitted, defaults to ``tile_size``.
        batch_size: Maximum number of image-prompt pairs to score together.
        threshold: Score threshold or per-prompt thresholds for score-map
            modes. Defaults to ``0.5``.
        confidence_threshold: Instance confidence threshold used by
            ``instance_only`` and ``fused`` score maps.
        save_scores: Whether to persist raw score maps for score-map prompts.
        save_score_preview: Whether to save score heatmap previews for
            score-map prompts.

    Returns:
        tuple: (labels_df, masks_dict)
            - labels_df: DataFrame with columns 'path', 'prompt', 'ratio'
              (long format, one row per path-prompt combination). Score-map
              modes add columns such as ``mode`` and ``threshold``.
            - masks_dict: Dictionary mapping filepath to dict of prompt -> bool mask.
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS.copy()
    if mode is None:
        mode = DEFAULT_MODE
    mode_map = _normalise_modes(prompts, mode)
    threshold_map = _normalise_prompt_floats(prompts, threshold, default=0.5)
    confidence_map = _normalise_prompt_floats(
        prompts, confidence_threshold, default=0.5,
    )
    score_tile_size = int(tile_size or 0)
    score_tile_stride = int(tile_stride or score_tile_size or 0)
    score_batch_size = int(batch_size)

    # Generate colors for preview
    color_map = _generate_colors(prompts, colors)

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    resolved_device = _get_device(device)
    logger.info(f"Using device: {resolved_device}")

    # Strategy B: DataLoader for I/O prefetching
    filepaths = profile["path"].tolist()
    dataset = _ImageDataset(filepaths, resize)
    loader_kwargs: dict = dict(
        batch_size=1,
        num_workers=num_workers,
        collate_fn=_collate_passthrough,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    # Strategy C: async mask saving in background threads
    save_executor = ThreadPoolExecutor(max_workers=2) if output_dir is not None else None
    save_futures: list[Future] = []

    labels_data = []
    masks_dict = {}

    from tqdm import tqdm
    from PIL import Image

    model_bundle = _load_models(
        resolved_device, checkpoint_path=checkpoint_path,
    )
    instance_prompts = [
        prompt for prompt in prompts if mode_map[prompt] == "instance"
    ]
    score_prompts = [
        prompt for prompt in prompts if mode_map[prompt] != "instance"
    ]

    for filepath, img_bgr, resized_rgb, scale in tqdm(
        loader, desc="segment", total=len(dataset),
    ):
        if img_bgr is None:
            logger.warning(f"Failed to read image: {filepath}")
            for prompt in prompts:
                labels_data.append({
                    "path": filepath,
                    "prompt": prompt,
                    "ratio": 0.0,
                    "mode": mode_map[prompt],
                })
            continue

        mask_entry: dict[str, np.ndarray] = {}
        score_stats: dict[str, dict[str, float]] = {}

        if instance_prompts:
            mask_entry.update(_segment_single_image(
                resized_rgb, img_bgr.shape[:2], instance_prompts,
                model_bundle, scale,
            ))

        if score_prompts:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            score_maps = _sam3_tiled_scores(
                pil_img,
                score_prompts,
                model_bundle,
                resolved_device,
                score_tile_size,
                score_tile_stride,
                score_batch_size,
                mode_map,
                confidence_map,
            )
            for prompt in score_prompts:
                score = score_maps[prompt]
                mask_entry[prompt] = score >= threshold_map[prompt]
                score_stats[prompt] = {
                    "threshold": threshold_map[prompt],
                    "score_mean": float(score.mean()) if score.size else 0.0,
                    "score_max": float(score.max()) if score.size else 0.0,
                    "tile_size": float(score_tile_size),
                    "tile_stride": float(score_tile_stride),
                    "batch_size": float(score_batch_size),
                }
                if output_dir is not None:
                    if save_scores:
                        scores_dir = Path(output_dir) / "scores"
                        scores_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            scores_dir
                            / f"{Path(filepath).stem}_{prompt}_score.npz",
                            score=score,
                        )
                    if save_score_preview:
                        _save_score_preview(
                            filepath, score, mask_entry[prompt],
                            Path(output_dir) / "score_preview"
                            / f"{Path(filepath).stem}_{prompt}_score.jpg",
                        )

        mask_entry = _apply_prompt_priority(mask_entry, prompts, img_bgr.shape[:2])

        for prompt in prompts:
            stats = score_stats.get(prompt, {})
            labels_data.append({
                "path": filepath,
                "prompt": prompt,
                "ratio": _calculate_ratio(mask_entry[prompt]),
                "mode": mode_map[prompt],
                **stats,
            })

        if output_dir is not None and save_executor is not None:
            # Strategy C: submit save to background thread
            future = save_executor.submit(
                _save_masks, filepath, mask_entry, output_dir, color_map,
                original_img=img_bgr,
            )
            save_futures.append(future)
            # Backpressure: prevent unbounded memory from queued saves
            while len(save_futures) > 8:
                save_futures.pop(0).result()
        else:
            masks_dict[filepath] = mask_entry

    # Wait for remaining saves to complete
    for future in save_futures:
        future.result()
    if save_executor is not None:
        save_executor.shutdown(wait=True)

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
        occupied = np.zeros((h, w), dtype=bool)
        for prompt, mask in masks.items():
            color = color_map.get(prompt, (0, 255, 0))
            visible = mask & ~occupied
            colored[visible] = color
            occupied |= mask

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


def _save_score_preview(
    filepath: str,
    score: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save a diagnostic score heatmap preview for score-map tuning."""
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return
    norm = np.clip(score / max(float(score.max()), 1e-6), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = img_bgr.copy()
    overlay[mask] = (
        0.45 * overlay[mask] + 0.55 * np.array([255, 255, 255])
    ).astype(np.uint8)
    combined = np.hstack([img_bgr, heat, overlay])
    max_w = 2400
    if combined.shape[1] > max_w:
        scale = max_w / combined.shape[1]
        combined = cv2.resize(
            combined,
            (max_w, int(combined.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)


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
