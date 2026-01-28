"""
T079: Quickstart workflow verification test.

This test verifies that the workflow documented in docs/quickstart.md
works correctly and produces expected results.
"""

import json
import os

import cv2
import numpy as np
import pandas as pd
import pytest


class TestQuickstartWorkflow:
    """Verify the quickstart documentation workflow works correctly."""

    def test_quickstart_basic_workflow(self, tmp_path):
        """
        Test the basic workflow from quickstart guide.

        Steps tested:
        1. Ingest images
        2. Profile images
        3. Segment
        4. Pre-QC
        5. Align
        6. AOI
        7. Viz
        8. Export
        """
        import yamami

        # === Setup: Create synthetic test images ===
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create test images following the filename convention
        num_images = 10
        for i in range(num_images):
            # Create gradient image (sky at top, ground at bottom)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            for row in range(100):
                brightness = 200 - int(100 * row / 100)  # Brighter at top
                img[row, :, :] = brightness

            day = i + 1
            filename = f"MRD_N_NikL_RGB_2024050{day:01d}_0800.jpg"
            cv2.imwrite(str(images_dir / filename), img)

        # === Step 1: Ingest images from directory (from quickstart) ===
        index = yamami.ingest(str(images_dir))
        assert len(index) == num_images, "Should find all test images"

        # === Step 2: Profile images ===
        profile = yamami.profile(index)
        assert len(profile) == num_images
        assert "mean_luma" in profile.columns
        assert "site" in profile.columns

        # === Step 3: Segment and filter ===
        segment_profile = profile.rename(columns={"path": "filepath"})
        labels, masks = yamami.segment(segment_profile)
        assert len(labels) == num_images

        # === Step 4: Pre-QC ===
        qc_result = yamami.pre_qc(segment_profile, labels)
        assert "is_usable" in qc_result.columns

        # Filter to usable images
        usable_mask = qc_result["is_usable"]
        usable_profile = segment_profile[usable_mask].reset_index(drop=True)

        # Need at least some usable images to continue
        if len(usable_profile) == 0:
            pytest.skip("No usable images after QC (test data issue)")

        # === Step 5: Align images ===
        ref_image = usable_profile["filepath"].iloc[0]
        segments, warp_params, aligned_dir = yamami.align(
            usable_profile,
            ref=ref_image,
            output_dir=str(output_dir / "aligned")
        )

        assert os.path.exists(aligned_dir)
        assert isinstance(segments, pd.DataFrame)
        assert isinstance(warp_params, pd.DataFrame)

        # === Step 6: Extract AOI ===
        skyline, aoi_mask = yamami.aoi(ref_image, output_dir=str(output_dir / "aoi"))

        assert isinstance(skyline, pd.DataFrame)
        assert "x" in skyline.columns
        assert "y" in skyline.columns
        assert isinstance(aoi_mask, np.ndarray)

        # === Step 7: Visualization ===
        # Create mock phenology data for visualization
        pheno_data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        (output_dir / "phenology").mkdir(parents=True, exist_ok=True)
        pheno_data.to_csv(output_dir / "phenology" / "phenology.csv", index=False)

        viz_result = yamami.viz(pheno_data, output_dir=str(output_dir / "viz"))
        assert "gup" in viz_result
        assert os.path.exists(viz_result["gup"])

        # === Step 8: Export ===
        export_path = yamami.export(str(output_dir))
        assert os.path.isdir(export_path)
        assert os.path.exists(os.path.join(export_path, "processing_log.json"))

    def test_quickstart_phenology_workflow(self, tmp_path):
        """Test the phenology extraction workflow from quickstart."""
        from yamami.phenology import phenology

        # Create synthetic GR time series as shown in quickstart
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        doy = dates.dayofyear

        # Simulate seasonal GR pattern
        winter_base = 0.32
        summer_peak = 0.48
        amplitude = summer_peak - winter_base

        # Double sigmoid pattern
        gup_mid = 120
        gdown_mid = 280
        gup_rate = 0.08
        gdown_rate = 0.06

        rising = 1 / (1 + np.exp(-gup_rate * (doy - gup_mid)))
        falling = 1 / (1 + np.exp(-gdown_rate * (doy - gdown_mid)))
        gr_values = winter_base + amplitude * (rising - falling)

        gr_data = pd.DataFrame({
            "date": dates,
            "doy": doy,
            "mean_gr": gr_values,
        })

        # Extract phenology
        pheno_result = phenology(gr_data)

        assert isinstance(pheno_result, pd.DataFrame)
        assert "gup_doy" in pheno_result.columns
        assert "gdown_doy" in pheno_result.columns
        assert "gmax_doy" in pheno_result.columns
        assert "fit_status" in pheno_result.columns

    def test_quickstart_snow_workflow(self, tmp_path):
        """Test the snow detection workflow from quickstart."""
        from yamami.snow import snow, snowmelt

        # Create profile with RGB values as shown in quickstart
        snow_profile = pd.DataFrame({
            "filepath": ["/path/img1.jpg", "/path/img2.jpg", "/path/img3.jpg"],
            "doy": [100, 130, 160],
            "R_mean": [240.0, 180.0, 100.0],
            "G_mean": [240.0, 180.0, 100.0],
            "B_mean": [240.0, 180.0, 100.0],
        })

        # Create AOI mask
        aoi_mask = np.ones((10, 10), dtype=np.uint8)

        # Detect snow
        snow_masks, thresholds = snow(snow_profile, aoi_mask)

        assert len(snow_masks) == 3
        assert len(thresholds) == 3

        # Estimate snowmelt
        snowmelt_df = snowmelt(snow_masks, aoi_mask)

        assert isinstance(snowmelt_df, pd.DataFrame)
        assert "row" in snowmelt_df.columns
        assert "col" in snowmelt_df.columns
        assert "doy" in snowmelt_df.columns
        assert "confidence" in snowmelt_df.columns
        assert "status" in snowmelt_df.columns

    def test_quickstart_filename_convention(self, tmp_path):
        """Test the filename convention documented in quickstart."""
        from yamami import parse_filename

        # Test the example filename from quickstart
        filename = "MRD_N_NikL_RGB_20240501_0800.jpg"
        result = parse_filename(filename)

        assert result is not None
        assert result["site"] == "MRD"
        assert result["azimuth"] == "N"
        assert result["camera"] == "NikL"
        assert result["band"] == "RGB"
        assert result["timestamp"].year == 2024
        assert result["timestamp"].month == 5
        assert result["timestamp"].day == 1
        assert result["timestamp"].hour == 8
        assert result["timestamp"].minute == 0

    def test_quickstart_viz_workflow(self, tmp_path):
        """Test the visualization workflow from quickstart."""
        from yamami import viz

        # Create phenology data as shown in quickstart
        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        output_dir = tmp_path / "viz"

        # Generate visualizations
        result = viz(data, output_dir=str(output_dir))

        # Verify output as documented
        assert "gup" in result
        assert "gdown" in result
        assert "gmax" in result

        # Verify files exist
        assert os.path.exists(result["gup"])
        assert os.path.exists(result["gdown"])
        assert os.path.exists(result["gmax"])

    def test_quickstart_export_workflow(self, tmp_path):
        """Test the export workflow from quickstart."""
        from yamami import export

        # Setup output structure as would be created by pipeline
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [100, 110]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        (output_dir / "viz").mkdir()
        # Create a dummy viz file
        np.ones((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(output_dir / "viz" / "gup.png"),
                    np.ones((10, 10, 3), dtype=np.uint8))

        # Export
        export_path = export(str(output_dir))

        # Verify structure as documented
        assert os.path.isdir(export_path)
        assert os.path.exists(os.path.join(export_path, "processing_log.json"))

        # Verify processing log contents
        with open(os.path.join(export_path, "processing_log.json")) as f:
            log_data = json.load(f)

        # As documented, should contain version, timestamp, file hashes
        assert "yamami_version" in log_data
        assert "timestamp" in log_data
        assert "file_hashes" in log_data
