import logging
import sys
from typing import Optional

LOG_FORMAT = "{levelname:<8} {asctime} - {filename}:{lineno} - {message}"


# Define color codes
class LogColors:
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors based on log level"""

    FORMATS = {
        logging.DEBUG: LogColors.GREY + LOG_FORMAT + LogColors.RESET,
        logging.INFO: LogColors.BLUE + LOG_FORMAT + LogColors.RESET,
        logging.WARNING: LogColors.YELLOW + LOG_FORMAT + LogColors.RESET,
        logging.ERROR: LogColors.RED + LOG_FORMAT + LogColors.RESET,
        logging.CRITICAL: LogColors.BOLD_RED + LOG_FORMAT + LogColors.RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S", style="{")
        return formatter.format(record)


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
) -> None:
    """
    Setup basic logging configuration for the application.

    Args:
        log_file: Path to the log file. If None, only console logging is
            configured.
        log_level: The base logging level (default: logging.INFO)
    """
    # Setup console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Setup file handler if log_file is specified (without colors)
    if log_file:
        file_formatter = logging.Formatter(
            LOG_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
            style="{",
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


class RawColoredFormatter(logging.Formatter):
    """Custom formatter for raw logs with subtle color"""

    def format(self, record):
        return f"{LogColors.GREY}{record.getMessage()}{LogColors.RESET}"


def get_raw_logger(name: Optional[str] = "raw_logger") -> logging.Logger:
    """Create a raw logger for unformatted logs with subtle coloring."""
    raw_logger = logging.getLogger(name)
    raw_logger.propagate = False
    if not raw_logger.hasHandlers():
        raw_handler = logging.StreamHandler()
        raw_handler.setFormatter(RawColoredFormatter())
        raw_logger.addHandler(raw_handler)

    return raw_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified name.

    Args:
        name: The name of the logger (typically __name__)

    Returns:
        logging.Logger: A configured logger instance
    """
    return logging.getLogger(name)
