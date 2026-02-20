"""
Integration tests using real image data from test_data_micro.

These tests use actual mountain time-lapse images to verify the
pipeline works correctly with real-world data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yamami
from yamami import ingest, parse_filename


ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_MICRO = ROOT / "test_data_micro"
TEST_DATA_FULL = ROOT / "test_data_full"


def _resolve_device() -> str:
    """Resolve device for GPU operations."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@pytest.fixture(scope="module")
def output_dir():
    """Test output directory for persisting results."""
    out = ROOT / "test_outputs" / "real_data"
    out.mkdir(parents=True, exist_ok=True)
    return out


class TestIngestProfile:
    """Tests for ingest on real data."""

    @pytest.mark.skipif(not TEST_DATA_MICRO.exists(), reason="test_data_micro not found")
    def test_ingest_finds_all_images(self):
        """ingest() should find all 15 images in test_data_micro."""
        df = ingest(str(TEST_DATA_MICRO))

        assert len(df) == 15
        assert "path" in df.columns

        for path in df["path"]:
            assert Path(path).exists()

    @pytest.mark.skipif(not TEST_DATA_MICRO.exists(), reason="test_data_micro not found")
    def test_ingest_extracts_metadata(self, output_dir):
        """ingest() should extract metadata from all images."""
        df = ingest(str(TEST_DATA_MICRO))

        df.to_csv(output_dir / "profile.csv", index=False)

        required_columns = [
            "path", "filename", "timestamp",
            "width", "height", "filesize",
            "mean_luma", "std_luma", "p05_luma", "p50_luma", "p95_luma",
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        assert df["timestamp"].notna().all(), "Some timestamps are missing"

        for ts in df["timestamp"]:
            assert ts.year == 2020, f"Unexpected year: {ts.year}"

    @pytest.mark.skipif(not TEST_DATA_MICRO.exists(), reason="test_data_micro not found")
    def test_ingest_luminance_values(self):
        """ingest() should compute valid luminance statistics."""
        df = ingest(str(TEST_DATA_MICRO))

        luma_cols = ["mean_luma", "std_luma", "p05_luma", "p50_luma", "p95_luma"]
        for col in luma_cols:
            assert df[col].notna().all(), f"{col} has missing values"
            assert (df[col] >= 0).all(), f"{col} has negative values"
            assert (df[col] <= 255).all(), f"{col} has values > 255"


class TestFilenameParser:
    """Tests for filename parsing on real data."""

    @pytest.mark.skipif(not TEST_DATA_MICRO.exists(), reason="test_data_micro not found")
    def test_parse_real_filenames(self):
        """parse_filename() should parse test_data_micro filenames."""
        df = ingest(str(TEST_DATA_MICRO))

        for _, row in df.iterrows():
            filename = Path(row["path"]).name
            result = parse_filename(filename)

            assert result is not None, f"Failed to parse: {filename}"
            assert result.year == 2020


class TestFullWorkflow:
    """Full pipeline integration test on real data."""

    @pytest.mark.skipif(not TEST_DATA_MICRO.exists(), reason="test_data_micro not found")
    def test_full_pipeline(self, output_dir):
        """Test complete pipeline: ingest -> segment -> ridge_qc -> align -> aoi -> gr -> phenology -> snow -> snowmelt."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA is not available (SAM3 requires CUDA)")
        except ImportError:
            pytest.skip("torch is not installed")

        try:
            import sam3  # noqa: F401
        except ImportError:
            pytest.skip("sam3 is not installed")

        # Step 1: Ingest (includes profiling)
        df = yamami.ingest(str(TEST_DATA_MICRO))
        assert len(df) == 15

        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_csv(output_dir / "profile.csv", index=False)

        # Step 2: Segmentation (returns long format)
        seg_dir = output_dir / "segment"
        seg_dir.mkdir(parents=True, exist_ok=True)

        labels_long, masks_dict = yamami.segment(
            df,
            prompts=["fog", "cloud", "sky", "sun", "reflection", "snow", "mountain"],
            output_dir=str(seg_dir),
            device="cuda",
        )
        labels_long.to_csv(seg_dir / "labels.csv", index=False)

        assert "path" in labels_long.columns
        assert "prompt" in labels_long.columns
        assert "ratio" in labels_long.columns

        # Merge segment ratios into profile (long -> wide)
        pivot = labels_long.pivot(
            index="path", columns="prompt", values="ratio"
        ).reset_index()
        pivot.columns = ["path"] + [f"{col}_ratio" for col in pivot.columns[1:]]
        df = df.merge(pivot, on="path", how="left")

        # Step 3: Ridge QC (basic QC + ridgeline detection + shift segmentation)
        target_image = df["path"].iloc[len(df) // 2]  # Use middle image as target
        masks_dir = seg_dir / "masks"

        df = yamami.ridge_qc(
            df,
            masks_dir=str(masks_dir),
            target_image=target_image,
            output_dir=str(output_dir),
        )

        assert "is_usable" in df.columns
        assert "reason_codes" in df.columns
        assert "ridge_match_success" in df.columns
        assert "segment_id" in df.columns

        df.to_csv(output_dir / "ridge_qc.csv", index=False)

        # Step 4: Alignment
        df = yamami.align(
            df,
            target_image=target_image,
            masks_dir=str(masks_dir),
            method="superpoint-lightglue",
            output_dir=str(output_dir),
            device=_resolve_device(),
            max_rounds=3,
            ridge_validation=True,
        )

        assert "align_status" in df.columns
        assert "align_rmse" in df.columns

        aligned_dir = output_dir / "aligned"
        assert aligned_dir.exists()

        df.to_csv(output_dir / "aligned_result.csv", index=False)

        # Step 5: AOI Extraction (from aoi_mask.png saved by align)
        aoi_mask_path = aligned_dir / "aoi_mask.png"
        assert aoi_mask_path.exists()
        aoi_mask = yamami.load_aoi_mask(str(aoi_mask_path))

        assert isinstance(aoi_mask, np.ndarray)
        assert aoi_mask.dtype == bool
        assert aoi_mask.sum() > 0

        # Step 6: GR + Phenology
        gr_out = output_dir / "gr"

        # Use aligned images for GR calculation
        aligned_df = df[df["align_status"] == "success"].copy()
        if aligned_df.empty:
            aligned_df = df.copy()

        gr_images, gr_ts = yamami.gr(
            aligned_df,
            aoi_mask > 0,
            output_dir=str(gr_out),
        )
        gr_ts.to_csv(gr_out / "gr_timeseries.csv", index=False)

        assert isinstance(gr_ts, pd.DataFrame)
        assert "date" in gr_ts.columns
        assert "doy" in gr_ts.columns
        assert "mean_gr" in gr_ts.columns

        if not gr_ts.empty:
            pheno = yamami.phenology(gr_ts)
            pheno_out = output_dir / "phenology"
            pheno_out.mkdir(parents=True, exist_ok=True)
            pheno.to_csv(pheno_out / "phenology.csv", index=False)

            assert "fit_status" in pheno.columns

        # Step 7: Snow + Snowmelt
        snow_out = output_dir / "snow"
        snow_masks, snow_stats = yamami.snow(
            aligned_df,
            aligned_dir=str(aligned_dir),
            masks_dir=str(masks_dir),
            mask=aoi_mask,
            output_dir=str(snow_out),
        )
        snow_stats.to_csv(snow_out / "snow_stats.csv", index=False)

        assert len(snow_masks) > 0

        snowmelt_out = output_dir / "snowmelt"
        snowmelt_df = yamami.snowmelt(
            snow_masks,
            aoi_mask,
            output_dir=str(snowmelt_out),
        )

        if not snowmelt_df.empty:
            snowmelt_df.to_csv(snowmelt_out / "snowmelt.csv", index=False)
            assert "doy" in snowmelt_df.columns
            assert "status" in snowmelt_df.columns

        # Step 8: Visualization
        viz_out = output_dir / "viz"
        if not gr_ts.empty:
            pheno_spatial = pd.DataFrame({
                "row": [0, 0, 1, 1],
                "col": [0, 1, 0, 1],
                "gup": [100, 110, 105, 115],
                "gdown": [250, 260, 255, 265],
                "gmax": [175, 185, 180, 190],
            })
            viz_result = yamami.viz(pheno_spatial, output_dir=str(viz_out))

            assert "gup" in viz_result
            assert Path(viz_result["gup"]).exists()

        # Step 9: Export
        export_result = yamami.export(str(output_dir))

        assert Path(export_result).exists()
        assert (Path(export_result) / "processing_log.json").exists()
