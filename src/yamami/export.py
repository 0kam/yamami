"""
Export module for YAMAMI.

Provides functionality to organize and export processing results
with documentation and provenance tracking.
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import yamami


def export(
    output_dir: str,
    export_dir: Optional[str] = None,
    include_log: bool = True,
) -> str:
    """
    Organize and export all results.

    Creates a structured export directory with all processing outputs,
    optionally including a processing log with version information,
    timestamps, and file hashes.

    Args:
        output_dir: Source directory containing processing outputs.
            Expected subdirectories: images/, phenology/, snowmelt/, viz/
        export_dir: Target directory for export. If None, creates
            'export' subdirectory alongside output_dir.
        include_log: Whether to create processing_log.json (default: True)

    Returns:
        str: Path to the export directory.

    Note:
        Export directory structure::

            export/
            +-- images/
            |   +-- profile_qc.csv
            +-- phenology/
            |   +-- gup.png, gdown.png, phenology.csv
            +-- snowmelt/
            |   +-- doy_map.png, snowmelt.csv
            +-- viz/
            |   +-- (PNG files)
            +-- processing_log.json

    Example:
        >>> export_path = export("/path/to/output")
        >>> print(export_path)
        /path/to/export
    """
    output_path = Path(output_dir)

    # Determine export directory
    if export_dir is None:
        export_path = output_path.parent / "export"
    else:
        export_path = Path(export_dir)

    # Create export directory
    export_path.mkdir(parents=True, exist_ok=True)

    # Track exported files for hashing
    exported_files = {}

    # Copy each subdirectory if it exists
    subdirs = ["images", "phenology", "snowmelt", "viz"]

    for subdir in subdirs:
        src_dir = output_path / subdir
        if src_dir.exists() and src_dir.is_dir():
            dst_dir = export_path / subdir
            _copy_directory(src_dir, dst_dir, exported_files)

    # Create processing log if requested
    if include_log:
        log_data = _create_processing_log(export_path, exported_files)
        log_path = export_path / "processing_log.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

    return str(export_path)


def _copy_directory(
    src_dir: Path,
    dst_dir: Path,
    exported_files: dict,
) -> None:
    """
    Copy directory contents, tracking files for hashing.

    Args:
        src_dir: Source directory
        dst_dir: Destination directory
        exported_files: Dict to populate with file paths and hashes
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        src_path = item
        dst_path = dst_dir / item.name

        if item.is_file():
            shutil.copy2(src_path, dst_path)

            # Calculate file hash
            file_hash = _calculate_hash(dst_path)
            relative_path = str(dst_path.name)
            exported_files[f"{dst_dir.name}/{relative_path}"] = file_hash

        elif item.is_dir():
            # Recursively copy subdirectories
            _copy_directory(item, dst_path, exported_files)


def _calculate_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        str: Hex-encoded SHA256 hash
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def _create_processing_log(
    export_path: Path,
    exported_files: dict,
) -> dict:
    """
    Create processing log with provenance information.

    Args:
        export_path: Path to export directory
        exported_files: Dict of file paths to hashes

    Returns:
        dict: Processing log data
    """
    log_data = {
        "yamami_version": yamami.__version__,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "export_path": str(export_path),
        },
        "file_hashes": exported_files,
    }

    return log_data
