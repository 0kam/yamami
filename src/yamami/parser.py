"""
Filename parser for yamami image files.

Extracts datetime from filenames using configurable patterns.
"""

import os
import re
from datetime import datetime
from typing import Optional

# Default patterns for datetime extraction (priority order)
DEFAULT_PATTERNS = [
    # Pattern 1: YYYYMMDD_HHMM (e.g., 20200420_1805)
    re.compile(r"(?P<date>\d{8})_(?P<time>\d{4})"),
    # Pattern 2: YYYY-MM-DD_HH-MM-SS (e.g., 2020-04-20_18-05-00)
    re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})"),
    # Pattern 3: YYYY-MM-DD_HH-MM (e.g., 2020-04-20_18-05)
    re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_(?P<hour>\d{2})-(?P<minute>\d{2})"),
    # Pattern 4: YYYYMMDD only (e.g., 20200420, time defaults to 00:00)
    re.compile(r"(?P<date>\d{8})(?!\d)"),
]


def parse_filename(
    filename: Optional[str],
    pattern: Optional[str] = None,
) -> Optional[datetime]:
    """
    Extract datetime from a filename.

    Args:
        filename: The filename to parse. Can include full path (basename
                  will be extracted automatically). None or empty returns None.
        pattern: Optional custom regex pattern with named groups.
                 Supported group names:
                 - 'date' + 'time': YYYYMMDD and HHMM format
                 - 'year', 'month', 'day', 'hour', 'minute', 'second': Individual components
                 If not provided, auto-detects common patterns.

    Returns:
        datetime or None: Extracted timestamp, or None if parsing fails.

    Default patterns (priority order):
        1. YYYYMMDD_HHMM
        2. YYYY-MM-DD_HH-MM-SS
        3. YYYY-MM-DD_HH-MM
        4. YYYYMMDD only (time defaults to 00:00)
    """
    if filename is None or not filename:
        return None

    # Extract basename if full path is provided
    basename = os.path.basename(filename)

    # Use custom pattern if provided
    if pattern is not None:
        try:
            regex = re.compile(pattern)
        except re.error:
            return None
        patterns_to_try = [regex]
    else:
        patterns_to_try = DEFAULT_PATTERNS

    for regex in patterns_to_try:
        match = regex.search(basename)
        if match:
            groups = match.groupdict()
            try:
                dt = _extract_datetime(groups)
                if dt is not None:
                    return dt
            except (ValueError, KeyError):
                continue

    return None


def _extract_datetime(groups: dict) -> Optional[datetime]:
    """
    Extract datetime from regex match groups.

    Supports two formats:
    - 'date' (YYYYMMDD) and optional 'time' (HHMM)
    - Individual components: 'year', 'month', 'day', 'hour', 'minute'

    Args:
        groups: Dictionary of named groups from regex match.

    Returns:
        datetime or None: Extracted datetime if valid, None otherwise.
    """
    if "date" in groups:
        # Format: YYYYMMDD (+ optional HHMM)
        date_str = groups["date"]
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])

        if "time" in groups and groups["time"]:
            time_str = groups["time"]
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
        else:
            hour = 0
            minute = 0

    elif "year" in groups:
        # Format: individual components
        year = int(groups["year"])
        month = int(groups["month"])
        day = int(groups["day"])
        hour = int(groups.get("hour", 0) or 0)
        minute = int(groups.get("minute", 0) or 0)

    else:
        return None

    # Validate and create datetime
    return datetime(year, month, day, hour, minute)
