"""
Contract tests for export and visualization functions.

These tests verify the public API contracts for viz() and export().
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestVizContract:
    """Contract tests for viz() function."""

    def test_viz_returns_dict(self, tmp_path):
        """viz() must return a dict of visualization arrays/paths."""
        from yamami.viz import viz

        # Create sample phenology data
        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        result = viz(data, output_dir=str(tmp_path))

        assert isinstance(result, dict)

    def test_viz_returns_paths_when_output_dir_provided(self, tmp_path):
        """viz() must return file paths when output_dir is specified."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        result = viz(data, output_dir=str(tmp_path))

        # All values should be paths (strings) that exist
        for key, value in result.items():
            assert isinstance(value, str), f"Expected path string for {key}"
            assert os.path.exists(value), f"File should exist: {value}"

    def test_viz_returns_arrays_when_no_output_dir(self):
        """viz() must return numpy arrays when no output_dir is specified."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        result = viz(data)

        # All values should be numpy arrays
        for key, value in result.items():
            assert isinstance(value, np.ndarray), f"Expected numpy array for {key}"

    def test_viz_phenology_data_keys(self, tmp_path):
        """viz() with phenology data must return gup, gdown, gmax keys."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        result = viz(data, output_dir=str(tmp_path))

        assert "gup" in result
        assert "gdown" in result
        assert "gmax" in result

    def test_viz_snowmelt_data_keys(self, tmp_path):
        """viz() with snowmelt data must return doy_map key."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "doy": [80, 85, 90, 95],
        })

        result = viz(data, output_dir=str(tmp_path))

        assert "doy_map" in result


class TestExportContract:
    """Contract tests for export() function."""

    def test_export_returns_string_path(self, tmp_path):
        """export() must return a string path."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create minimal expected structure
        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir))

        assert isinstance(result, str)
        assert os.path.isdir(result)

    def test_export_creates_directory_structure(self, tmp_path):
        """export() must create organized directory structure."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        export_dir = tmp_path / "export"

        # Create minimal output structure
        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir), export_dir=str(export_dir))

        # Verify directory structure
        assert os.path.isdir(result)
        assert os.path.isdir(os.path.join(result, "phenology"))

    def test_export_creates_processing_log(self, tmp_path):
        """export() must create processing_log.json."""
        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create minimal output structure
        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir), include_log=True)

        log_path = os.path.join(result, "processing_log.json")
        assert os.path.exists(log_path)

    def test_export_log_contains_required_fields(self, tmp_path):
        """export() processing_log.json must contain required fields."""
        import json

        from yamami.export import export

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create minimal output structure
        (output_dir / "phenology").mkdir()
        pd.DataFrame({"gup": [1, 2]}).to_csv(
            output_dir / "phenology" / "phenology.csv", index=False
        )

        result = export(str(output_dir), include_log=True)

        log_path = os.path.join(result, "processing_log.json")
        with open(log_path) as f:
            log_data = json.load(f)

        assert "yamami_version" in log_data
        assert "timestamp" in log_data
        assert "parameters" in log_data
        assert "file_hashes" in log_data
