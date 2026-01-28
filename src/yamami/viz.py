"""
Visualization module for YAMAMI.

Provides color-coded DOY (Day of Year) visualizations for phenology
and snowmelt data.
"""

import os
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use non-interactive backend for server environments
matplotlib.use("Agg")


def viz(
    data: Union[pd.DataFrame, dict],
    colormap: str = "viridis",
    vmin: Optional[int] = None,
    vmax: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Generate color-coded DOY visualizations.

    Creates matplotlib figures with colorbar/legend for phenology or
    snowmelt data. For phenology data (gup, gdown, gmax), generates
    three maps. For snowmelt data (doy), generates a single doy_map.

    Args:
        data: Input data as DataFrame or dict.
            For DataFrame: columns 'row', 'col', and data columns
            (gup, gdown, gmax for phenology; doy for snowmelt).
            For dict: keys are data names, values are 2D numpy arrays.
        colormap: Matplotlib colormap name (default: "viridis")
        vmin: Minimum value for colormap normalization
        vmax: Maximum value for colormap normalization
        output_dir: Directory to save PNG files. If None, returns arrays.

    Returns:
        dict: If output_dir is provided, returns dict of file paths.
              Otherwise, returns dict of numpy arrays (RGBA images).

    Example:
        >>> data = pd.DataFrame({
        ...     "row": [0, 0, 1, 1],
        ...     "col": [0, 1, 0, 1],
        ...     "gup": [100, 110, 105, 115],
        ...     "gdown": [250, 260, 255, 265],
        ...     "gmax": [175, 185, 180, 190],
        ... })
        >>> result = viz(data, output_dir="/path/to/output")
        >>> print(result.keys())
        dict_keys(['gup', 'gdown', 'gmax'])
    """
    # Determine data type and extract arrays
    if isinstance(data, dict):
        arrays = data
    else:
        arrays = _dataframe_to_arrays(data)

    # Determine what type of data we have
    data_keys = list(arrays.keys())
    is_phenology = any(k in data_keys for k in ["gup", "gdown", "gmax"])
    is_snowmelt = "doy" in data_keys or "doy_map" in data_keys

    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    result = {}

    if is_phenology:
        # Generate phenology visualizations
        for key in ["gup", "gdown", "gmax"]:
            if key in arrays:
                array = arrays[key]
                output = _create_visualization(
                    array,
                    title=_get_title(key),
                    colormap=colormap,
                    vmin=vmin,
                    vmax=vmax,
                    output_dir=output_dir,
                    filename=f"{key}.png",
                )
                result[key] = output

    if is_snowmelt:
        # Generate snowmelt visualization
        doy_key = "doy" if "doy" in arrays else "doy_map"
        array = arrays[doy_key]
        output = _create_visualization(
            array,
            title="Snowmelt DOY",
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            output_dir=output_dir,
            filename="doy_map.png",
        )
        result["doy_map"] = output

    return result


def _dataframe_to_arrays(df: pd.DataFrame) -> dict:
    """
    Convert DataFrame with row/col indices to 2D arrays.

    Args:
        df: DataFrame with 'row', 'col' columns and data columns

    Returns:
        dict: Data column names mapped to 2D numpy arrays
    """
    if "row" not in df.columns or "col" not in df.columns:
        raise ValueError("DataFrame must have 'row' and 'col' columns")

    rows = df["row"].values
    cols = df["col"].values

    # Determine array dimensions
    max_row = int(rows.max()) + 1
    max_col = int(cols.max()) + 1

    # Get data columns (exclude row/col)
    data_cols = [c for c in df.columns if c not in ["row", "col"]]

    result = {}
    for col_name in data_cols:
        # Create array filled with NaN
        array = np.full((max_row, max_col), np.nan)

        # Fill in values
        valid_mask = ~df[col_name].isna()
        valid_rows = rows[valid_mask].astype(int)
        valid_cols = cols[valid_mask].astype(int)
        array[valid_rows, valid_cols] = df.loc[valid_mask, col_name].values

        result[col_name] = array

    return result


def _get_title(key: str) -> str:
    """Get display title for data key."""
    titles = {
        "gup": "Green-up DOY",
        "gdown": "Green-down DOY",
        "gmax": "Maximum Greenness DOY",
    }
    return titles.get(key, key.upper())


def _create_visualization(
    array: np.ndarray,
    title: str,
    colormap: str,
    vmin: Optional[int],
    vmax: Optional[int],
    output_dir: Optional[str],
    filename: str,
) -> Union[str, np.ndarray]:
    """
    Create a single visualization with colorbar.

    Args:
        array: 2D numpy array of data
        title: Plot title
        colormap: Matplotlib colormap name
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        output_dir: Output directory (None for array output)
        filename: Output filename

    Returns:
        File path if output_dir provided, otherwise RGBA array
    """
    # Create figure with colorbar
    fig, ax = plt.subplots(figsize=(8, 6))

    # Handle vmin/vmax
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)

    # Create colormap with masked array for NaN handling
    masked_array = np.ma.masked_invalid(array)

    # Plot image
    im = ax.imshow(masked_array, cmap=colormap, vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Day of Year (DOY)")

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    if output_dir is not None:
        # Save to file
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path
    else:
        # Return as array
        fig.canvas.draw()
        # Get RGBA array from figure
        width, height = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape(height, width, 4)
        plt.close(fig)
        return rgba.copy()
