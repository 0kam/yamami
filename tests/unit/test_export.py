"""
Unit tests for yamami export module.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestExportDirectoryOrganization:
    """Tests for export directory organization."""

    def test_export_creates_images_subdir(self, tmp_path):
        """export() should create images/ subdirectory."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create images dir with profile_qc.csv
        (output_dir / "images").mkdir()
        pd.DataFrame({"qc": [1, 2]}).to_csv(
            output_dir / "images" / "profile_qc.csv", index=False
        )

        result = export(str(output_dir))

        images_dir = os.path.join(result, "images")
        assert os.path.isdir(images_dir)

    def test_export_creates_phenology_subdir(self, tmp_path):
        """export() should create phenology/ subdirectory."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir))

        phenology_dir = os.path.join(result, "phenology")
        assert os.path.isdir(phenology_dir)

    def test_export_creates_snowmelt_subdir(self, tmp_path):
        """export() should create snowmelt/ subdirectory if present."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "snowmelt").mkdir()
        pd.DataFrame({"doy": [80, 85]}).to_csv(
            output_dir / "snowmelt" / "snowmelt.csv", index=False
        )

        result = export(str(output_dir))

        snowmelt_dir = os.path.join(result, "snowmelt")
        assert os.path.isdir(snowmelt_dir)

    def test_export_creates_viz_subdir(self, tmp_path):
        """export() should create viz/ subdirectory if present."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "viz").mkdir()
        # Create dummy png file
        with open(output_dir / "viz" / "test.png", "wb") as f:
            f.write(b"PNG")

        result = export(str(output_dir))

        viz_dir = os.path.join(result, "viz")
        assert os.path.isdir(viz_dir)

    def test_export_copies_files(self, tmp_path):
        """export() should copy files to export directory."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "phenology").mkdir()
        df = pd.DataFrame({"gup": [1, 2, 3], "gdown": [4, 5, 6]})
        df.to_csv(output_dir / "phenology" / "phenology.csv", index=False)

        result = export(str(output_dir))

        exported_file = os.path.join(result, "phenology", "phenology.csv")
        assert os.path.exists(exported_file)

        # Verify content is correct
        loaded = pd.read_csv(exported_file)
        assert len(loaded) == 3

    def test_export_custom_export_dir(self, tmp_path):
        """export() should use custom export_dir if specified."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        export_dir = tmp_path / "my_export"

        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir), export_dir=str(export_dir))

        assert result == str(export_dir)
        assert os.path.isdir(export_dir)


class TestExportProcessingLog:
    """Tests for processing log generation."""

    def test_export_creates_processing_log(self, tmp_path):
        """export() should create processing_log.json."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir))

        log_path = os.path.join(result, "processing_log.json")
        assert os.path.exists(log_path)

    def test_export_log_contains_version(self, tmp_path):
        """export() log should contain yamami version."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir))

        log_path = os.path.join(result, "processing_log.json")
        with open(log_path) as f:
            log_data = json.load(f)

        assert "yamami_version" in log_data
        assert isinstance(log_data["yamami_version"], str)

    def test_export_log_contains_timestamp(self, tmp_path):
        """export() log should contain timestamp."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir))

        log_path = os.path.join(result, "processing_log.json")
        with open(log_path) as f:
            log_data = json.load(f)

        assert "timestamp" in log_data
        # Verify timestamp format (ISO 8601)
        from datetime import datetime
        datetime.fromisoformat(log_data["timestamp"])

    def test_export_log_contains_parameters(self, tmp_path):
        """export() log should contain parameters."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir))

        log_path = os.path.join(result, "processing_log.json")
        with open(log_path) as f:
            log_data = json.load(f)

        assert "parameters" in log_data
        assert isinstance(log_data["parameters"], dict)

    def test_export_log_contains_file_hashes(self, tmp_path):
        """export() log should contain file hashes."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir))

        log_path = os.path.join(result, "processing_log.json")
        with open(log_path) as f:
            log_data = json.load(f)

        assert "file_hashes" in log_data
        assert isinstance(log_data["file_hashes"], dict)
        # Should have at least one file hash
        assert len(log_data["file_hashes"]) >= 1

    def test_export_log_can_be_disabled(self, tmp_path):
        """export() should not create log when include_log=False."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir), include_log=False)

        log_path = os.path.join(result, "processing_log.json")
        assert not os.path.exists(log_path)


class TestExportEdgeCases:
    """Edge case tests for export."""

    def test_export_empty_output_dir(self, tmp_path):
        """export() should handle empty output directory."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = export(str(output_dir))

        assert os.path.isdir(result)

    def test_export_overwrites_existing_export(self, tmp_path):
        """export() should overwrite existing export directory."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        # Create old file
        (export_dir / "old_file.txt").write_text("old")

        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir), export_dir=str(export_dir))

        assert os.path.isdir(result)

    def test_export_preserves_file_content(self, tmp_path):
        """export() should preserve exact file content."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "phenology").mkdir()
        original_df = pd.DataFrame({
            "gup": [100, 110, 120],
            "gdown": [250, 260, 270],
            "gmax": [175, 185, 195],
        })
        original_df.to_csv(output_dir / "phenology" / "phenology.csv", index=False)

        result = export(str(output_dir))

        exported_df = pd.read_csv(os.path.join(result, "phenology", "phenology.csv"))
        pd.testing.assert_frame_equal(original_df, exported_df)

    def test_export_default_export_dir_name(self, tmp_path):
        """export() should create 'export' subdirectory by default."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir))

        # Default should be output_dir/../export or output_dir/export
        assert "export" in result
