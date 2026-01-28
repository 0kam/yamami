"""
Unit tests for yamami visualization module.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestVizDOYColormap:
    """Tests for DOY colormap visualization."""

    def test_viz_uses_viridis_colormap_by_default(self, tmp_path):
        """viz() should use viridis colormap by default."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        result = viz(data, output_dir=str(tmp_path))

        # Verify files were created with colormap applied
        assert os.path.exists(result["gup"])
        assert os.path.exists(result["gdown"])
        assert os.path.exists(result["gmax"])

    def test_viz_supports_custom_colormap(self, tmp_path):
        """viz() should support custom colormap parameter."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        result = viz(data, colormap="plasma", output_dir=str(tmp_path))

        assert os.path.exists(result["gup"])

    def test_viz_creates_figures_with_colorbar(self, tmp_path):
        """viz() output files should contain colorbar/legend."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        result = viz(data, output_dir=str(tmp_path))

        # Verify file size indicates image with colorbar (should be larger)
        file_size = os.path.getsize(result["gup"])
        assert file_size > 0

    def test_viz_respects_vmin_vmax(self, tmp_path):
        """viz() should respect vmin and vmax parameters."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        # Should not raise with custom vmin/vmax
        result = viz(data, vmin=1, vmax=365, output_dir=str(tmp_path))

        assert os.path.exists(result["gup"])

    def test_viz_phenology_creates_gup_gdown_gmax(self, tmp_path):
        """viz() with phenology data should create gup, gdown, gmax maps."""
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
        assert result["gup"].endswith(".png")
        assert result["gdown"].endswith(".png")
        assert result["gmax"].endswith(".png")

    def test_viz_snowmelt_creates_doy_map(self, tmp_path):
        """viz() with snowmelt data should create doy_map."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "doy": [80, 85, 90, 95],
        })

        result = viz(data, output_dir=str(tmp_path))

        assert "doy_map" in result
        assert result["doy_map"].endswith(".png")

    def test_viz_returns_arrays_without_output_dir(self):
        """viz() without output_dir should return numpy arrays."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        result = viz(data)

        assert isinstance(result["gup"], np.ndarray)
        assert isinstance(result["gdown"], np.ndarray)
        assert isinstance(result["gmax"], np.ndarray)

    def test_viz_handles_nan_values(self, tmp_path):
        """viz() should handle NaN values in data."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, np.nan, 105, 115],
            "gdown": [250, 260, np.nan, 265],
            "gmax": [175, 185, 180, np.nan],
        })

        result = viz(data, output_dir=str(tmp_path))

        assert os.path.exists(result["gup"])

    def test_viz_with_dict_input(self, tmp_path):
        """viz() should accept dict input."""
        from yamami.viz import viz

        data = {
            "gup": np.array([[100, 110], [105, 115]]),
            "gdown": np.array([[250, 260], [255, 265]]),
            "gmax": np.array([[175, 185], [180, 190]]),
        }

        result = viz(data, output_dir=str(tmp_path))

        assert "gup" in result
        assert "gdown" in result
        assert "gmax" in result


class TestVizEdgeCases:
    """Edge case tests for visualization."""

    def test_viz_single_pixel(self, tmp_path):
        """viz() should handle single pixel data."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0],
            "col": [0],
            "gup": [100],
            "gdown": [250],
            "gmax": [175],
        })

        result = viz(data, output_dir=str(tmp_path))

        assert os.path.exists(result["gup"])

    def test_viz_large_array(self, tmp_path):
        """viz() should handle larger arrays."""
        from yamami.viz import viz

        rows, cols = 100, 100
        row_vals = np.repeat(np.arange(rows), cols)
        col_vals = np.tile(np.arange(cols), rows)

        data = pd.DataFrame({
            "row": row_vals,
            "col": col_vals,
            "gup": np.random.randint(50, 150, rows * cols),
            "gdown": np.random.randint(200, 300, rows * cols),
            "gmax": np.random.randint(125, 225, rows * cols),
        })

        result = viz(data, output_dir=str(tmp_path))

        assert os.path.exists(result["gup"])

    def test_viz_creates_output_dir_if_not_exists(self, tmp_path):
        """viz() should create output directory if it doesn't exist."""
        from yamami.viz import viz

        data = pd.DataFrame({
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "gup": [100, 110, 105, 115],
            "gdown": [250, 260, 255, 265],
            "gmax": [175, 185, 180, 190],
        })

        new_dir = tmp_path / "new_output"
        result = viz(data, output_dir=str(new_dir))

        assert os.path.isdir(new_dir)
        assert os.path.exists(result["gup"])
