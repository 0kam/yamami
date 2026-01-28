"""
Integration test for full workflow on real image data.

This test runs the end-to-end pipeline on test_data_* datasets and
persists intermediate outputs under test_outputs/real_data for inspection.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import yamami


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "micro": ROOT / "test_data_micro",
    "mini": ROOT / "test_data_mini",
    "full": ROOT / "test_data_full",
}


def _selected_datasets() -> list[str]:
    raw = os.environ.get("YAMAMI_REAL_DATA", "micro")
    names = [name.strip() for name in raw.split(",") if name.strip()]
    return [name for name in names if name in DATASETS]


def _resolve_device() -> str:
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


@pytest.fixture(scope="session")
def output_root() -> Path:
    out = ROOT / "test_outputs" / "real_data"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.mark.real_data
@pytest.mark.parametrize("dataset_name", _selected_datasets())
def test_real_data_full_workflow(dataset_name: str, output_root: Path) -> None:
    data_dir = DATASETS[dataset_name]
    if not data_dir.exists():
        pytest.skip(f"{data_dir} not found")
    try:
        import sam3  # noqa: F401
    except ImportError:
        pytest.skip("sam3 is not installed (required for real_data workflow)")
    try:
        import torch
    except ImportError:
        pytest.skip("torch is not installed (required for real_data workflow)")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available (SAM3 requires a CUDA GPU)")

    max_images_raw = os.environ.get("YAMAMI_REAL_DATA_MAX_IMAGES", "0").strip()
    max_images = int(max_images_raw) if max_images_raw else 0
    snow_downsample_raw = os.environ.get("YAMAMI_REAL_DATA_SNOW_DOWNSAMPLE", "128")
    snow_downsample = int(snow_downsample_raw) if snow_downsample_raw else 128

    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: ingest and profile
    index = yamami.ingest(str(data_dir))
    assert len(index) > 0, "No images found in real data directory"
    index.to_csv(output_dir / "index.csv", index=False)

    profile = yamami.profile(index)
    profile = profile.sort_values("timestamp").reset_index(drop=True)
    profile.to_csv(output_dir / "profile.csv", index=False)

    workflow = profile.rename(columns={"path": "filepath"})
    workflow["timestamp"] = pd.to_datetime(workflow["timestamp"], errors="coerce")

    if max_images > 0:
        workflow = workflow.head(max_images).reset_index(drop=True)

    # Step 2: segmentation (masks)
    masks_dir = output_dir / "segment" / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    labels, _ = yamami.segment(
        workflow,
        prompts=["sky", "cloud", "fog", "snow"],
        output_dir=str(masks_dir),
        device="cuda",
    )
    (output_dir / "segment").mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_dir / "segment" / "labels.csv", index=False)
    assert any(masks_dir.glob("*.png")), "Segmentation masks were not saved"

    # Step 3: pre-QC
    qc_result = yamami.pre_qc(workflow, labels)
    (output_dir / "qc").mkdir(parents=True, exist_ok=True)
    qc_result.to_csv(output_dir / "qc" / "pre_qc.csv", index=False)

    usable_profile = workflow[qc_result["is_usable"]].reset_index(drop=True)
    if usable_profile.empty:
        usable_profile = workflow.copy()

    # Step 4: alignment (superpoint-lightglue)
    ref_image = usable_profile["filepath"].iloc[0]
    align_out = output_dir / "align"
    aligned_images = align_out / "aligned"
    segments, warp_params, aligned_dir = yamami.align(
        usable_profile,
        ref=ref_image,
        method="superpoint-lightglue",
        device=_resolve_device(),
        output_dir=str(aligned_images),
    )
    align_out.mkdir(parents=True, exist_ok=True)
    segments.to_csv(align_out / "segments.csv", index=False)
    warp_params.to_csv(align_out / "warp_params.csv", index=False)
    assert Path(aligned_dir).exists(), "Aligned directory not created"

    aligned_profile = usable_profile.copy()
    aligned_profile["filepath"] = aligned_profile["filepath"].apply(
        lambda p: str(Path(aligned_dir) / Path(p).name)
    )
    aligned_profile.to_csv(align_out / "aligned_profile.csv", index=False)

    # Step 5: AOI extraction
    aoi_out = output_dir / "aoi"
    skyline, aoi_mask = yamami.aoi(ref_image, output_dir=str(aoi_out))
    assert isinstance(skyline, pd.DataFrame)
    assert isinstance(aoi_mask, np.ndarray)

    # Step 6: GR + phenology
    gr_out = output_dir / "gr"
    gr_images, gr_ts = yamami.gr(
        aligned_profile[aligned_profile["timestamp"].notna()],
        aoi_mask > 0,
        output_dir=str(gr_out),
    )
    gr_out.mkdir(parents=True, exist_ok=True)
    gr_ts.to_csv(gr_out / "gr_timeseries.csv", index=False)

    if not gr_ts.empty:
        pheno = yamami.phenology(gr_ts)
        pheno_out = output_dir / "phenology"
        pheno_out.mkdir(parents=True, exist_ok=True)
        pheno.to_csv(pheno_out / "phenology.csv", index=False)

    # Step 7: snow + snowmelt (downsampled AOI to keep output size manageable)
    snow_profile = _compute_rgb_means(aligned_profile)
    if not snow_profile.empty:
        snow_aoi = _downsample_mask(aoi_mask, snow_downsample)
        snow_out = output_dir / "snow"
        snow_masks, thresholds = yamami.snow(
            snow_profile,
            snow_aoi,
            output_dir=str(snow_out),
        )
        assert not thresholds.empty

        snowmelt_out = output_dir / "snowmelt"
        snowmelt_df = yamami.snowmelt(
            snow_masks,
            snow_aoi,
            output_dir=str(snowmelt_out),
        )
        assert not snowmelt_df.empty
