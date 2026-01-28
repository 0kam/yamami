"""
Unit tests for yamami filename parser.
"""

from datetime import datetime

import pytest

from yamami.parser import parse_filename


class TestParseValidFilename:
    """Tests for parsing valid filenames."""

    def test_parse_yyyymmdd_hhmm_format(self):
        """Test parsing YYYYMMDD_HHMM format (most common)."""
        result = parse_filename("mrd_085_eos_vis_20200420_1805_R.JPG")
        assert result["timestamp"] == datetime(2020, 4, 20, 18, 5)

    def test_parse_standard_underscore_datetime(self):
        """Test parsing with underscored datetime."""
        result = parse_filename("image_20240501_0800.jpg")
        assert result["timestamp"] == datetime(2024, 5, 1, 8, 0)
        assert result["site"] is None

    def test_parse_yyyy_mm_dd_hh_mm_format(self):
        """Test parsing YYYY-MM-DD_HH-MM format."""
        result = parse_filename("image_2020-04-20_18-05.jpg")
        assert result["timestamp"] == datetime(2020, 4, 20, 18, 5)

    def test_parse_yyyymmdd_only_format(self):
        """Test parsing YYYYMMDD only format (time defaults to 00:00)."""
        result = parse_filename("photo_20200420.jpg")
        assert result["timestamp"] == datetime(2020, 4, 20, 0, 0)

    def test_parse_midnight_time(self):
        """Test parsing filename with midnight time."""
        result = parse_filename("photo_20240101_0000.jpg")
        assert result["timestamp"] == datetime(2024, 1, 1, 0, 0)

    def test_parse_end_of_day_time(self):
        """Test parsing filename with end of day time."""
        result = parse_filename("photo_20241231_2359.jpg")
        assert result["timestamp"] == datetime(2024, 12, 31, 23, 59)


class TestParseInvalidFilename:
    """Tests for handling invalid filenames."""

    def test_parse_empty_filename(self):
        """Test parsing empty filename returns None."""
        result = parse_filename("")
        assert result is None

    def test_parse_none_input(self):
        """Test parsing None input returns None."""
        result = parse_filename(None)
        assert result is None

    def test_parse_no_datetime_pattern(self):
        """Test parsing filename without datetime returns None."""
        result = parse_filename("random_file_name.jpg")
        assert result is None

    def test_parse_invalid_date_values(self):
        """Test parsing with invalid date values (e.g., month 13)."""
        result = parse_filename("photo_20241301_0800.jpg")
        assert result is None

    def test_parse_invalid_time_values(self):
        """Test parsing with invalid time values falls back to date-only pattern."""
        # Invalid time (25:00) in pattern 1 falls back to pattern 3 (date only)
        result = parse_filename("photo_20240501_2500.jpg")
        # Falls back to YYYYMMDD only pattern, time defaults to 00:00
        assert result["timestamp"] == datetime(2024, 5, 1, 0, 0)

    def test_parse_non_leap_year_feb_29(self):
        """Test parsing with Feb 29 in non-leap year returns None."""
        result = parse_filename("photo_20230229_0800.jpg")
        assert result is None


class TestEdgeCases:
    """Tests for edge cases in filename parsing."""

    def test_parse_with_full_path(self):
        """Test parsing with full path extracts basename."""
        result = parse_filename("/path/to/images/photo_20200420_1805.jpg")
        assert result["timestamp"] == datetime(2020, 4, 20, 18, 5)

    def test_parse_leap_year_date(self):
        """Test parsing with leap year date (Feb 29)."""
        result = parse_filename("photo_20240229_0800.jpg")
        assert result["timestamp"] == datetime(2024, 2, 29, 8, 0)

    def test_parse_different_extensions(self):
        """Test parsing with various file extensions."""
        extensions = ["jpg", "jpeg", "png", "tif", "tiff", "JPG", "JPEG"]
        for ext in extensions:
            result = parse_filename(f"photo_20200420_1805.{ext}")
            assert result["timestamp"] == datetime(2020, 4, 20, 18, 5), f"Failed for .{ext}"

    def test_parse_real_data_filename(self):
        """Test parsing actual test_data_micro filename."""
        result = parse_filename("mrd_085_eos_vis_20200420_1805_R.JPG")
        assert result["timestamp"] == datetime(2020, 4, 20, 18, 5)
        assert result["site"] == "mrd"
        assert result["azimuth"] == "085"
        assert result["camera"] == "eos"
        assert result["band"] == "vis"

    def test_parse_multiple_date_patterns(self):
        """Test that first valid pattern is used."""
        # Has both YYYYMMDD_HHMM and a date that looks like YYYYMMDD
        result = parse_filename("img_20200420_1805_v2.jpg")
        assert result["timestamp"] == datetime(2020, 4, 20, 18, 5)


class TestCustomPattern:
    """Tests for custom pattern parsing."""

    def test_custom_pattern_with_t_separator(self):
        """Test custom pattern with T separator."""
        result = parse_filename(
            "IMG_20200420T1805.jpg",
            pattern=r"(?P<date>\d{8})T(?P<time>\d{4})",
        )
        assert result["timestamp"] == datetime(2020, 4, 20, 18, 5)

    def test_custom_pattern_individual_components(self):
        """Test custom pattern with individual datetime components."""
        result = parse_filename(
            "photo-2020.04.20-18.05.jpg",
            pattern=r"(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})-(?P<hour>\d{2})\.(?P<minute>\d{2})",
        )
        assert result["timestamp"] == datetime(2020, 4, 20, 18, 5)

    def test_custom_pattern_date_only(self):
        """Test custom pattern with date only."""
        result = parse_filename(
            "photo_DATE20200420.jpg",
            pattern=r"DATE(?P<date>\d{8})",
        )
        assert result["timestamp"] == datetime(2020, 4, 20, 0, 0)

    def test_invalid_custom_pattern(self):
        """Test invalid regex pattern returns None."""
        result = parse_filename("photo.jpg", pattern=r"[invalid(regex")
        assert result is None

    def test_custom_pattern_no_match(self):
        """Test custom pattern that doesn't match returns None."""
        result = parse_filename(
            "photo_20200420_1805.jpg",
            pattern=r"NONEXISTENT(?P<date>\d{8})",
        )
        assert result is None
