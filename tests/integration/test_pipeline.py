"""
Integration tests for the image collection and profiling pipeline.
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytest

from yamami import ingest, profile


class TestIngestProfilePipeline:
    """Integration tests for ingest -> profile pipeline (T021)."""

    def test_complete_workflow(self, tmp_path):
        """Test complete ingest -> profile workflow with synthetic images."""
        # Create synthetic test images with valid yamami filenames
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create test images with different characteristics
        test_cases = [
            ("MRD_N_NikL_RGB_20240501_0800.jpg", 100, 100, 128),  # Gray
            ("MRD_N_NikL_RGB_20240501_1200.jpg", 200, 150, 255),  # White
            ("MRD_S_NikL_NIR_20240502_0600.jpg", 150, 100, 0),  # Black
        ]

        for filename, width, height, brightness in test_cases:
            img = np.full((height, width, 3), brightness, dtype=np.uint8)
            cv2.imwrite(str(images_dir / filename), img)

        # Run ingest
        index = ingest(str(images_dir))

        # Verify ingest results
        assert len(index) == 3
        assert "path" in index.columns

        # Run profile
        result = profile(index)

        # Verify profile results
        assert len(result) == 3

        # Check all required columns exist
        required_columns = [
            "path",
            "filename",
            "site",
            "azimuth",
            "camera",
            "band",
            "timestamp",
            "width",
            "height",
            "filesize",
            "mean_luma",
            "std_luma",
            "p05_luma",
            "p50_luma",
            "p95_luma",
        ]
        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Verify metadata extraction works
        assert "MRD" in result["site"].values
        assert "N" in result["azimuth"].values or "S" in result["azimuth"].values

        # Verify image dimensions are correct
        widths = result["width"].tolist()
        heights = result["height"].tolist()
        assert 100 in widths
        assert 200 in widths
        assert 150 in widths or 100 in heights

        # Verify luminance stats are computed
        assert result["mean_luma"].notna().all()
        assert result["std_luma"].notna().all()

    def test_workflow_with_unparseable_filenames(self, tmp_path):
        """Test workflow handles images with unparseable filenames."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create images - some with valid, some with invalid filenames
        valid_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(images_dir / "MRD_N_NikL_RGB_20240501_0800.jpg"), valid_img)

        invalid_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(images_dir / "random_filename.jpg"), invalid_img)

        # Run pipeline
        index = ingest(str(images_dir))
        result = profile(index)

        # Both images should be processed
        assert len(result) == 2

        # Valid filename should have metadata
        valid_row = result[result["filename"] == "MRD_N_NikL_RGB_20240501_0800.jpg"]
        assert len(valid_row) == 1
        assert valid_row.iloc[0]["site"] == "MRD"

        # Invalid filename should have None/NaN for metadata fields
        invalid_row = result[result["filename"] == "random_filename.jpg"]
        assert len(invalid_row) == 1
        assert pd.isna(invalid_row.iloc[0]["site"])

    def test_workflow_with_subdirectories(self, tmp_path):
        """Test workflow finds images in subdirectories."""
        images_dir = tmp_path / "images"
        subdir1 = images_dir / "site1"
        subdir2 = images_dir / "site2"
        subdir1.mkdir(parents=True)
        subdir2.mkdir(parents=True)

        # Create images in different directories
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(subdir1 / "MRD_N_NikL_RGB_20240501_0800.jpg"), img)
        cv2.imwrite(str(subdir2 / "ABC_S_NikL_RGB_20240502_1200.jpg"), img)

        # Run pipeline
        index = ingest(str(images_dir))
        result = profile(index)

        assert len(result) == 2

    def test_workflow_empty_directory(self, tmp_path):
        """Test workflow handles empty directory gracefully."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        index = ingest(str(empty_dir))
        result = profile(index)

        assert len(index) == 0
        assert len(result) == 0

    def test_workflow_preserves_path_information(self, tmp_path):
        """Test that absolute paths are preserved through the pipeline."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        image_path = images_dir / "MRD_N_NikL_RGB_20240501_0800.jpg"
        cv2.imwrite(str(image_path), img)

        index = ingest(str(images_dir))
        result = profile(index)

        # Path should be absolute and match the original
        assert str(image_path) == result.iloc[0]["path"]

    def test_workflow_computes_reasonable_luminance(self, tmp_path):
        """Test that luminance values are reasonable for known images."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create a known gray image (brightness 128)
        gray_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(images_dir / "MRD_N_NikL_RGB_20240501_0800.jpg"), gray_img)

        index = ingest(str(images_dir))
        result = profile(index)

        # Mean luminance should be close to 128
        mean_luma = result.iloc[0]["mean_luma"]
        assert 125 < mean_luma < 131, f"Expected ~128, got {mean_luma}"

        # Std should be close to 0 for uniform image
        std_luma = result.iloc[0]["std_luma"]
        assert std_luma < 1, f"Expected ~0, got {std_luma}"


class TestGrPhenologyWorkflow:
    """Integration tests for gr() -> phenology() workflow (T064)."""

    def test_gr_to_phenology_with_synthetic_seasonal_data(self, tmp_path):
        """
        Test complete workflow from GR calculation to phenology extraction.

        Uses synthetic seasonal data to verify the full pipeline works.
        """
        from yamami.phenology import gr, phenology

        np.random.seed(42)

        # Generate 365 days of synthetic GR data with seasonal pattern
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        doy = np.arange(1, 366)

        # Simulate realistic seasonal GR pattern using double sigmoid
        winter_base = 0.32
        summer_peak = 0.48
        amplitude = summer_peak - winter_base

        gup_mid = 120  # Mid-point of green-up
        gup_rate = 0.08
        gdown_mid = 280  # Mid-point of green-down
        gdown_rate = 0.06

        # Calculate GR using double sigmoid formula
        rising = 1 / (1 + np.exp(-gup_rate * (doy - gup_mid)))
        falling = 1 / (1 + np.exp(-gdown_rate * (doy - gdown_mid)))
        gr_values = winter_base + amplitude * (rising - falling)

        # Add small noise and clip to valid range
        noise = 0.01 * np.random.randn(365)
        gr_values = np.clip(gr_values + noise, 0, 1)

        # Create GR timeseries DataFrame
        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        # Run phenology extraction
        pheno_result = phenology(gr_data)

        # Verify output structure
        assert isinstance(pheno_result, pd.DataFrame)
        assert "gup_doy" in pheno_result.columns
        assert "gdown_doy" in pheno_result.columns
        assert "gmax_doy" in pheno_result.columns
        assert "fit_status" in pheno_result.columns

        if pheno_result["fit_status"].iloc[0] == "success":
            # Verify phenology dates are reasonable
            gup = pheno_result["gup_doy"].iloc[0]
            gmax = pheno_result["gmax_doy"].iloc[0]
            gdown = pheno_result["gdown_doy"].iloc[0]

            # Green-up should be in spring (roughly DOY 80-160)
            assert 50 < gup < 200, f"Green-up DOY {gup} not in expected range"

            # Green-max should be in summer (roughly DOY 150-250)
            assert 100 < gmax < 300, f"Green-max DOY {gmax} not in expected range"

            # Green-down should be in autumn (roughly DOY 240-330)
            assert 200 < gdown < 365, f"Green-down DOY {gdown} not in expected range"

            # Order should be: gup < gmax < gdown
            assert gup < gmax < gdown, \
                f"Phenology dates out of order: gup={gup}, gmax={gmax}, gdown={gdown}"

    def test_gr_to_phenology_handles_missing_data(self, tmp_path):
        """Test workflow gracefully handles gaps in time series."""
        from yamami.phenology import phenology

        np.random.seed(42)

        # Create time series with gaps (missing some days)
        all_dates = pd.date_range("2024-01-01", periods=365, freq="D")
        all_doy = np.arange(1, 366)

        winter_base = 0.32
        summer_peak = 0.48
        amplitude = summer_peak - winter_base
        gup_mid = 120
        gdown_mid = 280
        gup_rate = 0.08
        gdown_rate = 0.06

        rising = 1 / (1 + np.exp(-gup_rate * (all_doy - gup_mid)))
        falling = 1 / (1 + np.exp(-gdown_rate * (all_doy - gdown_mid)))
        all_gr = winter_base + amplitude * (rising - falling)

        # Remove 30% of data randomly
        keep_mask = np.random.rand(365) > 0.3
        dates = all_dates[keep_mask]
        doy = all_doy[keep_mask]
        gr_values = all_gr[keep_mask]

        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        # Should not crash even with missing data
        pheno_result = phenology(gr_data)

        assert isinstance(pheno_result, pd.DataFrame)
        assert "fit_status" in pheno_result.columns

    def test_gr_output_can_feed_into_phenology(self, tmp_path):
        """Test that gr() output format is compatible with phenology() input."""
        from yamami.phenology import phenology

        # Mock gr() output format
        mock_timeseries = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=365, freq="D"),
            "doy": np.arange(1, 366),
            "mean_gr": 0.35 + 0.1 * np.sin(np.arange(1, 366) * 2 * np.pi / 365),
        })

        # Verify phenology() can accept this format
        result = phenology(mock_timeseries)

        assert isinstance(result, pd.DataFrame)
        assert "gup_doy" in result.columns


class TestVizExportWorkflow:
    """Integration tests for viz -> export workflow (T072)."""

    def test_viz_to_export_phenology_workflow(self, tmp_path):
        """Test complete viz -> export workflow for phenology data."""
        from yamami.export import export
        from yamami.viz import viz

        # Create phenology data
        rows, cols = 10, 10
        row_vals = np.repeat(np.arange(rows), cols)
        col_vals = np.tile(np.arange(cols), rows)

        phenology_data = pd.DataFrame({
            "row": row_vals,
            "col": col_vals,
            "gup": np.random.randint(80, 150, rows * cols),
            "gdown": np.random.randint(250, 300, rows * cols),
            "gmax": np.random.randint(150, 200, rows * cols),
        })

        # Step 1: Create output directory with phenology data
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "phenology").mkdir()
        phenology_data.to_csv(output_dir / "phenology" / "phenology.csv", index=False)

        # Step 2: Generate visualizations
        viz_dir = output_dir / "viz"
        viz_result = viz(phenology_data, output_dir=str(viz_dir))

        # Verify viz outputs
        assert "gup" in viz_result
        assert "gdown" in viz_result
        assert "gmax" in viz_result
        assert os.path.exists(viz_result["gup"])
        assert os.path.exists(viz_result["gdown"])
        assert os.path.exists(viz_result["gmax"])

        # Step 3: Export all results
        export_dir = tmp_path / "export"
        export_result = export(str(output_dir), export_dir=str(export_dir))

        # Verify export structure
        assert os.path.isdir(export_result)
        assert os.path.isdir(os.path.join(export_result, "phenology"))
        assert os.path.isdir(os.path.join(export_result, "viz"))
        assert os.path.exists(os.path.join(export_result, "processing_log.json"))

        # Verify exported files
        assert os.path.exists(os.path.join(export_result, "phenology", "phenology.csv"))
        assert os.path.exists(os.path.join(export_result, "viz", "gup.png"))
        assert os.path.exists(os.path.join(export_result, "viz", "gdown.png"))
        assert os.path.exists(os.path.join(export_result, "viz", "gmax.png"))

    def test_viz_to_export_snowmelt_workflow(self, tmp_path):
        """Test complete viz -> export workflow for snowmelt data."""
        from yamami.export import export
        from yamami.viz import viz

        # Create snowmelt data
        rows, cols = 10, 10
        row_vals = np.repeat(np.arange(rows), cols)
        col_vals = np.tile(np.arange(cols), rows)

        snowmelt_data = pd.DataFrame({
            "row": row_vals,
            "col": col_vals,
            "doy": np.random.randint(60, 120, rows * cols),
        })

        # Step 1: Create output directory with snowmelt data
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "snowmelt").mkdir()
        snowmelt_data.to_csv(output_dir / "snowmelt" / "snowmelt.csv", index=False)

        # Step 2: Generate visualizations
        viz_dir = output_dir / "viz"
        viz_result = viz(snowmelt_data, output_dir=str(viz_dir))

        # Verify viz outputs
        assert "doy_map" in viz_result
        assert os.path.exists(viz_result["doy_map"])

        # Step 3: Export all results
        export_result = export(str(output_dir))

        # Verify export structure
        assert os.path.isdir(export_result)
        assert os.path.isdir(os.path.join(export_result, "snowmelt"))
        assert os.path.isdir(os.path.join(export_result, "viz"))

    def test_export_processing_log_integrity(self, tmp_path):
        """Test that processing log contains accurate information."""
        import json

        from yamami.export import export
        from yamami.viz import viz

        # Create phenology data
        phenology_data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        # Setup output structure
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "phenology").mkdir()
        phenology_data.to_csv(output_dir / "phenology" / "phenology.csv", index=False)

        # Generate viz
        viz_dir = output_dir / "viz"
        viz(phenology_data, output_dir=str(viz_dir))

        # Export
        export_result = export(str(output_dir))

        # Load and verify processing log
        log_path = os.path.join(export_result, "processing_log.json")
        with open(log_path) as f:
            log_data = json.load(f)

        # Verify version
        import yamami
        assert log_data["yamami_version"] == yamami.__version__

        # Verify timestamp is valid ISO format
        from datetime import datetime as dt
        timestamp = dt.fromisoformat(log_data["timestamp"])
        assert timestamp is not None

        # Verify file hashes exist for exported files
        assert len(log_data["file_hashes"]) > 0

    def test_full_workflow_with_multiple_data_types(self, tmp_path):
        """Test workflow with both phenology and snowmelt data."""
        from yamami.export import export
        from yamami.viz import viz

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create phenology data
        phenology_data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })
        (output_dir / "phenology").mkdir()
        phenology_data.to_csv(output_dir / "phenology" / "phenology.csv", index=False)

        # Create snowmelt data
        snowmelt_data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "doy": [80, 85, 90, 95],
        })
        (output_dir / "snowmelt").mkdir()
        snowmelt_data.to_csv(output_dir / "snowmelt" / "snowmelt.csv", index=False)

        # Create images QC data
        (output_dir / "images").mkdir()
        pd.DataFrame({"qc_score": [0.9, 0.85, 0.92]}).to_csv(
            output_dir / "images" / "profile_qc.csv", index=False
        )

        # Generate visualizations for both
        viz_dir = output_dir / "viz"
        viz_dir.mkdir()

        viz(phenology_data, output_dir=str(viz_dir))
        viz(snowmelt_data, output_dir=str(viz_dir))

        # Export everything
        export_result = export(str(output_dir))

        # Verify complete export structure
        assert os.path.isdir(os.path.join(export_result, "images"))
        assert os.path.isdir(os.path.join(export_result, "phenology"))
        assert os.path.isdir(os.path.join(export_result, "snowmelt"))
        assert os.path.isdir(os.path.join(export_result, "viz"))

        # Verify all data files
        assert os.path.exists(os.path.join(export_result, "images", "profile_qc.csv"))
        assert os.path.exists(os.path.join(export_result, "phenology", "phenology.csv"))
        assert os.path.exists(os.path.join(export_result, "snowmelt", "snowmelt.csv"))

        # Verify all viz files
        assert os.path.exists(os.path.join(export_result, "viz", "gup.png"))
        assert os.path.exists(os.path.join(export_result, "viz", "gdown.png"))
        assert os.path.exists(os.path.join(export_result, "viz", "gmax.png"))
        assert os.path.exists(os.path.join(export_result, "viz", "doy_map.png"))


class TestSnowPipeline:
    """T053: Integration tests for snow -> snowmelt workflow."""

    def test_snow_to_snowmelt_workflow(self):
        """Test complete snow detection to snowmelt estimation pipeline."""
        from yamami.snow import snow, snowmelt

        # Create synthetic profile with seasonal snow pattern
        # Early season: full snow coverage
        # Mid season: decreasing snow
        # Late season: no snow
        profile_df = pd.DataFrame({
            "filepath": [
                "/path/early1.jpg",
                "/path/early2.jpg",
                "/path/mid1.jpg",
                "/path/mid2.jpg",
                "/path/late1.jpg",
                "/path/late2.jpg",
            ],
            "doy": [100, 105, 120, 130, 150, 160],
            "R_mean": [240.0, 235.0, 200.0, 160.0, 100.0, 90.0],
            "G_mean": [240.0, 235.0, 200.0, 160.0, 100.0, 90.0],
            "B_mean": [240.0, 235.0, 200.0, 160.0, 100.0, 90.0],
        })

        # Create AOI mask
        aoi_mask = np.ones((20, 20), dtype=np.uint8)

        # Step 1: Detect snow
        snow_masks, thresholds = snow(profile_df, aoi_mask)

        # Verify snow detection results
        assert len(snow_masks) == 6, "Should have mask for each date"
        assert len(thresholds) == 6, "Should have threshold for each image"

        # Step 2: Estimate snowmelt
        snowmelt_result = snowmelt(snow_masks, aoi_mask)

        # Verify snowmelt results
        assert isinstance(snowmelt_result, pd.DataFrame)
        assert "doy" in snowmelt_result.columns
        assert "row" in snowmelt_result.columns
        assert "col" in snowmelt_result.columns
        assert "confidence" in snowmelt_result.columns
        assert "status" in snowmelt_result.columns

    def test_pipeline_with_partial_aoi(self):
        """Test pipeline with partial AOI coverage."""
        from yamami.snow import snow, snowmelt

        profile_df = pd.DataFrame({
            "filepath": ["/path/img1.jpg", "/path/img2.jpg"],
            "doy": [100, 150],
            "R_mean": [240.0, 100.0],
            "G_mean": [240.0, 100.0],
            "B_mean": [240.0, 100.0],
        })

        # AOI covers only center region
        aoi_mask = np.zeros((20, 20), dtype=np.uint8)
        aoi_mask[5:15, 5:15] = 1

        snow_masks, _ = snow(profile_df, aoi_mask)
        snowmelt_result = snowmelt(snow_masks, aoi_mask)

        # All results should be within AOI bounds
        assert all(snowmelt_result["row"] >= 5)
        assert all(snowmelt_result["row"] < 15)
        assert all(snowmelt_result["col"] >= 5)
        assert all(snowmelt_result["col"] < 15)

    def test_pipeline_data_consistency(self):
        """Test that data flows correctly between snow and snowmelt."""
        from yamami.snow import snow, snowmelt

        profile_df = pd.DataFrame({
            "filepath": ["/path/img1.jpg", "/path/img2.jpg", "/path/img3.jpg"],
            "doy": [100, 120, 140],
            "R_mean": [240.0, 180.0, 100.0],
            "G_mean": [240.0, 180.0, 100.0],
            "B_mean": [240.0, 180.0, 100.0],
        })
        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        snow_masks, _ = snow(profile_df, aoi_mask)

        # Verify mask keys match profile DOY values
        expected_doys = set(profile_df["doy"].tolist())
        actual_doys = set(snow_masks.keys())
        assert expected_doys == actual_doys, \
            "Snow mask keys should match profile DOY values"

        # All masks should have same shape as AOI
        for doy, mask in snow_masks.items():
            assert mask.shape == aoi_mask.shape, \
                f"Mask for DOY {doy} should have same shape as AOI"

        # Run snowmelt
        snowmelt_result = snowmelt(snow_masks, aoi_mask)

        # All DOY values in result should be from the input DOYs
        valid_doys = set(snow_masks.keys())
        for doy in snowmelt_result["doy"]:
            if not pd.isna(doy):
                assert doy in valid_doys, \
                    f"Snowmelt DOY {doy} should be from input DOYs"


class TestQCAlignmentWorkflow:
    """Integration tests for segment -> pre_qc -> align -> aoi -> qc workflow (T043)."""

    def test_segment_to_pre_qc(self, tmp_path):
        """Test segment output flows into pre_qc."""
        from yamami.segment import segment
        from yamami.qc import pre_qc

        # Create test images
        profile = self._create_test_profile(tmp_path, num_images=3)

        # Step 1: Segment
        labels, masks = segment(profile)

        # Step 2: Pre-QC
        qc_result = pre_qc(profile, labels)

        # Verify outputs
        assert isinstance(qc_result, pd.DataFrame)
        assert "is_usable" in qc_result.columns
        assert len(qc_result) == len(profile)

    def test_pre_qc_to_align(self, tmp_path):
        """Test pre_qc filters images before align."""
        from yamami.segment import segment
        from yamami.qc import pre_qc
        from yamami.align import align

        profile = self._create_test_profile(tmp_path, num_images=5)

        # Step 1: Segment
        labels, masks = segment(profile)

        # Step 2: Pre-QC
        qc_result = pre_qc(profile, labels)

        # Filter to usable images
        usable_mask = qc_result["is_usable"]
        usable_profile = profile[usable_mask].reset_index(drop=True)

        if len(usable_profile) > 0:
            # Step 3: Align
            ref_image = usable_profile["filepath"].iloc[0]
            segments, warp_params, aligned_dir = align(
                usable_profile, ref=ref_image, output_dir=str(tmp_path / "aligned")
            )

            assert isinstance(segments, pd.DataFrame)
            assert isinstance(warp_params, pd.DataFrame)
            assert os.path.exists(aligned_dir)

    def test_align_to_aoi(self, tmp_path):
        """Test align output provides reference for aoi."""
        from yamami.align import align
        from yamami.aoi import aoi

        profile = self._create_test_profile(tmp_path, num_images=5)

        # Step 1: Align
        ref_image = profile["filepath"].iloc[0]
        segments, warp_params, aligned_dir = align(
            profile, ref=ref_image, output_dir=str(tmp_path / "aligned")
        )

        # Step 2: AOI (using reference image)
        skyline, aoi_mask = aoi(ref_image, output_dir=str(tmp_path / "aoi"))

        assert isinstance(skyline, pd.DataFrame)
        assert isinstance(aoi_mask, np.ndarray)
        assert "x" in skyline.columns
        assert "y" in skyline.columns

    def test_aoi_to_full_qc(self, tmp_path):
        """Test aoi output flows into full qc."""
        from yamami.segment import segment
        from yamami.aoi import aoi
        from yamami.qc import qc

        profile = self._create_test_profile(tmp_path, num_images=3)

        # Step 1: Segment
        labels, masks = segment(profile)

        # Step 2: AOI
        ref_image = profile["filepath"].iloc[0]
        skyline, aoi_mask = aoi(ref_image, output_dir=str(tmp_path / "aoi"))

        # Add aoi_cloud_ratio to labels (would normally be computed with masks)
        labels["aoi_cloud_ratio"] = 0.1  # Mock value

        # Step 3: Full QC
        qc_result = qc(profile, labels, aoi_mask)

        assert isinstance(qc_result, pd.DataFrame)
        assert "is_usable" in qc_result.columns
        assert "reason_codes" in qc_result.columns

    def test_full_qc_alignment_pipeline(self, tmp_path):
        """Test complete pipeline: segment -> pre_qc -> align -> aoi -> qc."""
        from yamami.segment import segment
        from yamami.qc import pre_qc, qc
        from yamami.align import align
        from yamami.aoi import aoi

        profile = self._create_test_profile(tmp_path, num_images=5)

        # Step 1: Segment
        labels, masks = segment(profile)
        assert isinstance(labels, pd.DataFrame)
        assert isinstance(masks, dict)

        # Step 2: Pre-QC
        pre_qc_result = pre_qc(profile, labels)
        assert "is_usable" in pre_qc_result.columns

        # Filter usable images
        usable_mask = pre_qc_result["is_usable"]
        usable_profile = profile[usable_mask].reset_index(drop=True)
        usable_labels = labels[usable_mask].reset_index(drop=True)

        if len(usable_profile) > 0:
            # Step 3: Align
            ref_image = usable_profile["filepath"].iloc[0]
            segments, warp_params, aligned_dir = align(
                usable_profile, ref=ref_image, output_dir=str(tmp_path / "aligned")
            )
            assert os.path.exists(aligned_dir)

            # Step 4: AOI
            skyline, aoi_mask = aoi(ref_image, output_dir=str(tmp_path / "aoi"))
            assert isinstance(aoi_mask, np.ndarray)

            # Add aoi_cloud_ratio (mock)
            usable_labels["aoi_cloud_ratio"] = 0.1

            # Step 5: Full QC
            full_qc_result = qc(usable_profile, usable_labels, aoi_mask)
            assert isinstance(full_qc_result, pd.DataFrame)
            assert "is_usable" in full_qc_result.columns

    def test_pipeline_handles_all_rejected(self, tmp_path):
        """Test pipeline handles case where all images are rejected by pre_qc."""
        from yamami.segment import segment
        from yamami.qc import pre_qc

        # Create very dark images (will be rejected)
        image_paths = []
        for i in range(3):
            img_path = tmp_path / f"dark_image_{i}.jpg"
            img = np.zeros((100, 100, 3), dtype=np.uint8)  # Black images
            cv2.imwrite(str(img_path), img)
            image_paths.append(str(img_path))

        profile = pd.DataFrame(
            {
                "filepath": image_paths,
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            }
        )

        # Segment
        labels, masks = segment(profile)

        # Pre-QC should reject all
        qc_result = pre_qc(profile, labels)

        # All should be rejected
        assert qc_result["is_usable"].sum() == 0

    def test_filepath_consistency_through_pipeline(self, tmp_path):
        """Test filepaths are consistent through pipeline."""
        from yamami.segment import segment
        from yamami.qc import pre_qc

        profile = self._create_test_profile(tmp_path, num_images=3)
        original_paths = set(profile["filepath"])

        labels, _ = segment(profile)
        qc_result = pre_qc(profile, labels)

        # All original paths should be in results
        assert set(labels["filepath"]) == original_paths
        assert set(qc_result["filepath"]) == original_paths

    def _create_test_profile(self, tmp_path, num_images=5):
        """Helper to create test profile DataFrame."""
        image_paths = []
        for i in range(num_images):
            img_path = tmp_path / f"test_image_{i}.jpg"
            # Create images with varying brightness
            brightness = 100 + i * 30  # 100, 130, 160, 190, 220
            img = np.ones((100, 100, 3), dtype=np.uint8) * brightness
            cv2.imwrite(str(img_path), img)
            image_paths.append(str(img_path))

        return pd.DataFrame(
            {
                "filepath": image_paths,
                "timestamp": pd.date_range("2024-01-01", periods=num_images, freq="h"),
                "site": ["MRD"] * num_images,
                "azimuth": ["N"] * num_images,
                "camera": ["NikL"] * num_images,
                "band": ["RGB"] * num_images,
            }
        )


class TestFullEndToEndPipeline:
    """T078: Full end-to-end integration test covering complete pipeline."""

    def test_complete_pipeline_ingest_to_export(self, tmp_path):
        """
        Test complete YAMAMI pipeline from ingest to export.

        This test exercises the full workflow:
        ingest -> profile -> segment -> pre_qc -> align -> aoi -> qc ->
        phenology -> viz -> export

        Uses synthetic images with seasonal brightness variation to simulate
        a real monitoring scenario.
        """
        import yamami
        from yamami import (
            ingest, profile, segment, pre_qc, align, aoi, qc,
            phenology, viz, export
        )

        # === Step 0: Create synthetic test data ===
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create 30 synthetic images simulating a month of monitoring
        # with varying brightness to simulate seasonal patterns
        np.random.seed(42)
        num_images = 30
        base_brightness = 120

        for i in range(num_images):
            # Simulate seasonal brightness variation
            day = i + 1
            brightness = int(base_brightness + 40 * np.sin(day * np.pi / 30))
            brightness = max(50, min(230, brightness))

            # Create image with gradient (sky brighter than ground)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            for row in range(100):
                # Sky is brighter (top), ground is darker (bottom)
                row_brightness = brightness + int(50 * (1 - row / 100))
                row_brightness = max(0, min(255, row_brightness))
                img[row, :, :] = row_brightness

            # Add some variation per image
            img = np.clip(img + np.random.randint(-10, 10, img.shape), 0, 255).astype(np.uint8)

            # Save with valid filename
            date_str = f"202405{day:02d}"
            filename = f"MRD_N_NikL_RGB_{date_str}_1200.jpg"
            cv2.imwrite(str(images_dir / filename), img)

        # === Step 1: Ingest ===
        index = ingest(str(images_dir))
        assert len(index) == num_images, f"Expected {num_images} images, got {len(index)}"

        # === Step 2: Profile ===
        profile_df = profile(index)
        assert len(profile_df) == num_images
        assert "mean_luma" in profile_df.columns
        assert profile_df["mean_luma"].notna().all()

        # Verify metadata parsing worked
        assert profile_df["site"].notna().any(), "Site should be parsed from filename"
        assert "MRD" in profile_df["site"].values

        # === Step 3: Segment ===
        # Prepare for segment (rename path to filepath)
        segment_profile = profile_df.rename(columns={"path": "filepath"})
        labels, masks = segment(segment_profile)

        assert len(labels) == num_images
        assert "filepath" in labels.columns
        assert "brightness" in labels.columns

        # === Step 4: Pre-QC ===
        qc1_result = pre_qc(segment_profile, labels)
        assert "is_usable" in qc1_result.columns
        assert "reason_codes" in qc1_result.columns

        # Some images should pass (we created varied brightness)
        usable_count = qc1_result["is_usable"].sum()
        assert usable_count > 0, "At least some images should pass pre-QC"

        # Filter to usable images
        usable_profile = segment_profile[qc1_result["is_usable"]].reset_index(drop=True)
        usable_labels = labels[qc1_result["is_usable"]].reset_index(drop=True)

        # === Step 5: Align ===
        ref_image = usable_profile["filepath"].iloc[0]
        aligned_dir = str(output_dir / "aligned")
        segments_df, warp_params, aligned_path = align(
            usable_profile,
            ref=ref_image,
            output_dir=aligned_dir
        )

        assert os.path.exists(aligned_path)
        assert isinstance(segments_df, pd.DataFrame)
        assert isinstance(warp_params, pd.DataFrame)

        # === Step 6: AOI ===
        aoi_dir = str(output_dir / "aoi")
        skyline, aoi_mask = aoi(ref_image, output_dir=aoi_dir)

        assert isinstance(skyline, pd.DataFrame)
        assert "x" in skyline.columns
        assert "y" in skyline.columns
        assert isinstance(aoi_mask, np.ndarray)
        assert aoi_mask.dtype == np.uint8

        # AOI mask should have some area
        aoi_area = (aoi_mask > 0).sum()
        assert aoi_area > 0, "AOI mask should cover some pixels"

        # === Step 7: Full QC ===
        usable_labels["aoi_cloud_ratio"] = 0.1  # Mock value
        qc2_result = qc(usable_profile, usable_labels, aoi_mask)

        assert "is_usable" in qc2_result.columns
        final_usable_count = qc2_result["is_usable"].sum()
        assert final_usable_count > 0, "Some images should pass full QC"

        # === Step 8: Phenology (using synthetic GR data) ===
        # Create synthetic GR time series
        dates = pd.date_range("2024-05-01", periods=num_images, freq="D")
        doy = dates.dayofyear

        # Simulate realistic GR pattern
        gr_values = 0.35 + 0.08 * np.sin((doy - 120) * np.pi / 180)
        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        pheno_result = phenology(gr_data)
        assert isinstance(pheno_result, pd.DataFrame)
        assert "fit_status" in pheno_result.columns

        # === Step 9: Create data for visualization ===
        (output_dir / "phenology").mkdir(exist_ok=True)
        (output_dir / "viz").mkdir(exist_ok=True)

        # Create mock phenology spatial data
        pheno_spatial = pd.DataFrame({
            "row": [0, 0, 1, 1, 2, 2],
            "col": [0, 1, 0, 1, 0, 1],
            "gup": [100, 105, 110, 108, 112, 115],
            "gdown": [250, 255, 260, 258, 262, 265],
            "gmax": [175, 180, 185, 183, 188, 190],
        })

        pheno_spatial.to_csv(output_dir / "phenology" / "phenology.csv", index=False)

        viz_result = viz(pheno_spatial, output_dir=str(output_dir / "viz"))

        assert "gup" in viz_result
        assert "gdown" in viz_result
        assert "gmax" in viz_result
        assert os.path.exists(viz_result["gup"])

        # === Step 10: Export ===
        export_dir = str(tmp_path / "export")
        export_result = export(str(output_dir), export_dir=export_dir)

        assert os.path.isdir(export_result)
        assert os.path.exists(os.path.join(export_result, "processing_log.json"))

        # Verify processing log
        with open(os.path.join(export_result, "processing_log.json")) as f:
            log_data = json.load(f)

        assert log_data["yamami_version"] == yamami.__version__
        assert "timestamp" in log_data
        assert "file_hashes" in log_data

        # === Verify overall pipeline integrity ===
        # Total image count should be preserved through pipeline
        assert len(index) == num_images
        assert len(profile_df) == num_images
        assert len(labels) == num_images
        assert len(qc1_result) == num_images

    def test_pipeline_with_empty_input(self, tmp_path):
        """Test pipeline handles empty input gracefully."""
        from yamami import ingest, profile

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        index = ingest(str(empty_dir))
        assert len(index) == 0

        profile_df = profile(index)
        assert len(profile_df) == 0

    def test_pipeline_module_imports(self):
        """Test all pipeline modules can be imported from main namespace."""
        import yamami

        # Verify all public API functions are accessible
        assert hasattr(yamami, "ingest")
        assert hasattr(yamami, "profile")
        assert hasattr(yamami, "segment")
        assert hasattr(yamami, "pre_qc")
        assert hasattr(yamami, "qc")
        assert hasattr(yamami, "align")
        assert hasattr(yamami, "aoi")
        assert hasattr(yamami, "snow")
        assert hasattr(yamami, "snowmelt")
        assert hasattr(yamami, "gr")
        assert hasattr(yamami, "phenology")
        assert hasattr(yamami, "viz")
        assert hasattr(yamami, "export")
        assert hasattr(yamami, "parse_filename")
        assert hasattr(yamami, "get_logger")

        # Verify version is accessible
        assert hasattr(yamami, "__version__")
        assert isinstance(yamami.__version__, str)

    def test_pipeline_data_flow_consistency(self, tmp_path):
        """Test data flows correctly between pipeline stages."""
        from yamami import ingest, profile, segment, pre_qc

        # Create test images
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        for i in range(5):
            img = np.ones((100, 100, 3), dtype=np.uint8) * (100 + i * 20)
            filename = f"MRD_N_NikL_RGB_2024050{i+1}_1200.jpg"
            cv2.imwrite(str(images_dir / filename), img)

        # Run pipeline
        index = ingest(str(images_dir))
        profile_df = profile(index)

        # Verify columns are consistent
        assert "path" in index.columns
        assert "path" in profile_df.columns

        # Rename for segment compatibility
        segment_profile = profile_df.rename(columns={"path": "filepath"})
        labels, masks = segment(segment_profile)

        # Verify filepath consistency
        assert set(segment_profile["filepath"]) == set(labels["filepath"])

        # Verify QC maintains filepaths
        qc_result = pre_qc(segment_profile, labels)
        assert set(segment_profile["filepath"]) == set(qc_result["filepath"])
