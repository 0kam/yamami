#!/usr/bin/env python3
"""
Run the YAMAMI workflow on real image data and persist step outputs.

Example:
  python examples/real_data_workflow.py \
      --data-dir test_data_micro \
      --output-dir test_outputs/real_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import yamami


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _downsample_mask(mask: np.ndarray, target: int) -> np.ndarray:
    if target <= 0:
        return mask
    resized = cv2.resize(mask, (target, target), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8) * 255


def _compute_rgb_means(profile: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in profile.iterrows():
        filepath = row["filepath"]
        timestamp = pd.to_datetime(row["timestamp"], errors="coerce")
        if pd.isna(timestamp):
            continue

        img = cv2.imread(str(filepath))
        if img is None:
            continue

        b_mean = float(img[:, :, 0].mean())
        g_mean = float(img[:, :, 1].mean())
        r_mean = float(img[:, :, 2].mean())
        records.append(
            {
                "filepath": filepath,
                "doy": int(timestamp.dayofyear),
                "R_mean": r_mean,
                "G_mean": g_mean,
                "B_mean": b_mean,
            }
        )

    return pd.DataFrame(records)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YAMAMI on real data.")
    parser.add_argument("--data-dir", required=True, help="Image directory path.")
    parser.add_argument(
        "--output-dir",
        default="test_outputs/real_run",
        help="Directory to store outputs.",
    )
    parser.add_argument(
        "--alignment-method",
        default="superpoint-lightglue",
        help="Alignment method (default: superpoint-lightglue).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for alignment/segmentation models.",
    )
    parser.add_argument(
        "--segment-resize",
        type=int,
        default=512,
        help="Segmentation resize (max side length).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit number of images (0 = all).",
    )
    parser.add_argument(
        "--snow-downsample",
        type=int,
        default=256,
        help="Downsample size for snowmelt AOI mask (0 = no downsample).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    _ensure_dir(output_dir)

    device = _resolve_device(args.device)
    if device != "cuda":
        raise SystemExit("SAM3 requires a CUDA-capable GPU (device='cuda').")
    try:
        import sam3  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Segmentation requires 'sam3'. Install from facebookresearch/sam3."
        ) from e
    if args.alignment_method.lower() not in ("akaze", "sift"):
        try:
            import imm  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "Alignment method requires 'image-matching-models' (imm). "
                "Install with: pip install yamami"
            ) from e

    # Step 1: ingest and profile
    index = yamami.ingest(str(data_dir))
    if index.empty:
        print(f"No images found under: {data_dir}")
        return 1
    index.to_csv(output_dir / "index.csv", index=False)

    profile = yamami.profile(index)
    profile = profile.sort_values("timestamp").reset_index(drop=True)
    profile["timestamp"] = pd.to_datetime(profile["timestamp"], errors="coerce")
    profile.to_csv(output_dir / "profile.csv", index=False)

    workflow = profile.rename(columns={"path": "filepath"})
    if args.max_images > 0:
        workflow = workflow.head(args.max_images).reset_index(drop=True)

    # Step 2: segmentation (masks)
    segment_dir = output_dir / "segment"
    masks_dir = segment_dir / "masks"
    _ensure_dir(masks_dir)
    labels, _ = yamami.segment(
        workflow,
        prompts=["sky", "cloud", "fog", "snow"],
        resize=args.segment_resize,
        device=device,
        output_dir=str(masks_dir),
    )
    _ensure_dir(segment_dir)
    labels.to_csv(segment_dir / "labels.csv", index=False)

    # Step 3: pre-QC
    qc_dir = output_dir / "qc"
    qc_result = yamami.pre_qc(workflow, labels)
    _ensure_dir(qc_dir)
    qc_result.to_csv(qc_dir / "pre_qc.csv", index=False)

    usable_profile = workflow[qc_result["is_usable"]].reset_index(drop=True)
    if usable_profile.empty:
        usable_profile = workflow.copy()

    # Step 4: alignment (superpoint-lightglue)
    ref_image = usable_profile["filepath"].iloc[0]
    align_dir = output_dir / "align"
    aligned_images = align_dir / "aligned"
    segments, warp_params, aligned_dir = yamami.align(
        usable_profile,
        ref=ref_image,
        method=args.alignment_method,
        device=device,
        output_dir=str(aligned_images),
    )
    _ensure_dir(align_dir)
    segments.to_csv(align_dir / "segments.csv", index=False)
    warp_params.to_csv(align_dir / "warp_params.csv", index=False)

    aligned_profile = usable_profile.copy()
    aligned_profile["filepath"] = aligned_profile["filepath"].apply(
        lambda p: str(Path(aligned_dir) / Path(p).name)
    )
    aligned_profile.to_csv(align_dir / "aligned_profile.csv", index=False)

    # Step 5: AOI extraction
    aoi_dir = output_dir / "aoi"
    skyline, aoi_mask = yamami.aoi(ref_image, output_dir=str(aoi_dir))
    skyline.to_csv(aoi_dir / "skyline.csv", index=False)

    # Step 6: GR + phenology
    gr_dir = output_dir / "gr"
    gr_images, gr_ts = yamami.gr(
        aligned_profile[aligned_profile["timestamp"].notna()],
        aoi_mask > 0,
        output_dir=str(gr_dir),
    )
    _ensure_dir(gr_dir)
    gr_ts.to_csv(gr_dir / "gr_timeseries.csv", index=False)

    if not gr_ts.empty:
        pheno_dir = output_dir / "phenology"
        pheno = yamami.phenology(gr_ts)
        _ensure_dir(pheno_dir)
        pheno.to_csv(pheno_dir / "phenology.csv", index=False)

    # Step 7: snow + snowmelt (downsampled AOI to keep output size manageable)
    snow_profile = _compute_rgb_means(aligned_profile)
    if not snow_profile.empty:
        snow_aoi = _downsample_mask(aoi_mask, args.snow_downsample)
        snow_dir = output_dir / "snow"
        snow_masks, thresholds = yamami.snow(
            snow_profile,
            snow_aoi,
            output_dir=str(snow_dir),
        )
        thresholds.to_csv(snow_dir / "snow_thresholds.csv", index=False)

        snowmelt_dir = output_dir / "snowmelt"
        snowmelt_df = yamami.snowmelt(
            snow_masks,
            snow_aoi,
            output_dir=str(snowmelt_dir),
        )
        snowmelt_df.to_csv(snowmelt_dir / "snowmelt.csv", index=False)

    print(f"Outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
