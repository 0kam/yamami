"""
Logging configuration for yamami.

Provides structured logging with consistent format across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Default log format with timestamp, level, logger name, and message
DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Get a configured logger for the specified module.

    Creates a logger with structured formatting that includes timestamp,
    log level, logger name, and message. By default, logs INFO and above
    to console. Optionally can log to a file as well.

    Args:
        name: The name for the logger (typically __name__ of the module).
        level: Console logging level. Defaults to INFO.
        log_file: Optional path to a log file. If provided, a file handler
            will be added.
        file_level: File logging level. Defaults to DEBUG.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        2024-01-15 10:30:45,123 - INFO - yamami.module - Processing started
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist to avoid duplicate handlers
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to allow all levels

        # Create formatter
        formatter = logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False

    return logger
