"""
Unit tests for yamami logging setup.
"""

import logging
import re
from io import StringIO

import pytest

from yamami.logging import get_logger


class TestGetLogger:
    """Tests for get_logger function."""

    def test_logger_initialization_with_module_name(self):
        """Test that logger is initialized with the correct module name."""
        logger = get_logger("yamami.test_module")
        assert logger.name == "yamami.test_module"
        assert isinstance(logger, logging.Logger)

    def test_logger_returns_same_instance_for_same_name(self):
        """Test that get_logger returns the same logger for the same name."""
        logger1 = get_logger("yamami.same_name")
        logger2 = get_logger("yamami.same_name")
        assert logger1 is logger2

    def test_logger_returns_different_instances_for_different_names(self):
        """Test that get_logger returns different loggers for different names."""
        logger1 = get_logger("yamami.module1")
        logger2 = get_logger("yamami.module2")
        assert logger1 is not logger2


class TestStructuredLogFormat:
    """Tests for structured log format."""

    def _capture_log_output(self, logger, level, message):
        """Helper to capture log output using StringIO."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)

        # Get the formatter from the logger's handlers
        if logger.handlers:
            formatter = logger.handlers[0].formatter
            if formatter:
                handler.setFormatter(formatter)

        logger.addHandler(handler)
        try:
            log_method = getattr(logger, level.lower())
            log_method(message)
            return stream.getvalue()
        finally:
            logger.removeHandler(handler)

    def test_log_format_includes_timestamp(self):
        """Test that log output includes timestamp."""
        logger = get_logger("yamami.format_test_ts")
        output = self._capture_log_output(logger, "INFO", "Test message")

        # Check for timestamp pattern: YYYY-MM-DD HH:MM:SS
        timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        assert re.search(timestamp_pattern, output)

    def test_log_format_includes_level(self):
        """Test that log output includes log level."""
        logger = get_logger("yamami.format_test_level")
        output = self._capture_log_output(logger, "INFO", "Test message")

        assert "INFO" in output

    def test_log_format_includes_message(self):
        """Test that log output includes the message."""
        logger = get_logger("yamami.format_test_msg")
        test_message = "This is a test log message"
        output = self._capture_log_output(logger, "INFO", test_message)

        assert test_message in output

    def test_log_format_includes_logger_name(self):
        """Test that log output includes the logger name."""
        logger_name = "yamami.format_test_name"
        logger = get_logger(logger_name)
        output = self._capture_log_output(logger, "INFO", "Test message")

        assert logger_name in output

    def test_formatted_output_structure(self):
        """Test that the formatted output has correct structure."""
        logger = get_logger("yamami.format_structure")

        # Create a StringIO to capture formatted output
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        # Get the formatter from the logger's handlers
        if logger.handlers:
            formatter = logger.handlers[0].formatter
            if formatter:
                handler.setFormatter(formatter)

        logger.addHandler(handler)

        try:
            logger.info("Test structured output")
            output = stream.getvalue()

            # Verify output contains expected components
            # Format: YYYY-MM-DD HH:MM:SS - LEVEL - name - message
            timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
            assert re.search(timestamp_pattern, output)
            assert "INFO" in output
            assert "yamami.format_structure" in output
            assert "Test structured output" in output
        finally:
            logger.removeHandler(handler)


class TestLogLevels:
    """Tests for log level functionality."""

    def _capture_log_output(self, logger, level, message, handler_level=logging.DEBUG):
        """Helper to capture log output using StringIO."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(handler_level)

        # Get the formatter from the logger's handlers
        if logger.handlers:
            formatter = logger.handlers[0].formatter
            if formatter:
                handler.setFormatter(formatter)

        logger.addHandler(handler)
        try:
            log_method = getattr(logger, level.lower())
            log_method(message)
            return stream.getvalue()
        finally:
            logger.removeHandler(handler)

    def test_debug_level_works(self):
        """Test that DEBUG level logging works."""
        logger = get_logger("yamami.level_debug")
        output = self._capture_log_output(logger, "DEBUG", "Debug message")

        assert "DEBUG" in output
        assert "Debug message" in output

    def test_info_level_works(self):
        """Test that INFO level logging works."""
        logger = get_logger("yamami.level_info")
        output = self._capture_log_output(logger, "INFO", "Info message")

        assert "INFO" in output
        assert "Info message" in output

    def test_warning_level_works(self):
        """Test that WARNING level logging works."""
        logger = get_logger("yamami.level_warning")
        output = self._capture_log_output(logger, "WARNING", "Warning message")

        assert "WARNING" in output
        assert "Warning message" in output

    def test_error_level_works(self):
        """Test that ERROR level logging works."""
        logger = get_logger("yamami.level_error")
        output = self._capture_log_output(logger, "ERROR", "Error message")

        assert "ERROR" in output
        assert "Error message" in output

    def test_default_level_filters_debug(self):
        """Test that default INFO level filters out DEBUG messages."""
        logger = get_logger("yamami.default_level")

        # Create a new handler with INFO level to simulate default behavior
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)

        logger.addHandler(handler)

        try:
            logger.debug("This should be filtered")
            output = stream.getvalue()
            assert "This should be filtered" not in output
        finally:
            logger.removeHandler(handler)

    def test_info_and_above_are_logged_by_default(self):
        """Test that INFO, WARNING, ERROR are logged by default."""
        logger = get_logger("yamami.level_defaults")

        # Test with INFO level handler (default behavior)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)

        if logger.handlers:
            formatter = logger.handlers[0].formatter
            if formatter:
                handler.setFormatter(formatter)

        logger.addHandler(handler)

        try:
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            output = stream.getvalue()

            assert "INFO" in output
            assert "Info message" in output
            assert "WARNING" in output
            assert "Warning message" in output
            assert "ERROR" in output
            assert "Error message" in output
        finally:
            logger.removeHandler(handler)
