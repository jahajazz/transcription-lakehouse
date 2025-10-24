"""
Logging configuration for the lakehouse package.

Provides centralized logging with configurable levels and handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "lakehouse",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up and configure a logger with console and/or file handlers.

    Args:
        name: Logger name (default: "lakehouse")
        level: Logging level as string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Optional path to log file. If provided, logs are written to this file.
        console_output: Whether to output logs to console (default: True)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("lakehouse", level="DEBUG", log_file=Path("lakehouse.log"))
        >>> logger.info("Processing started")
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "lakehouse") -> logging.Logger:
    """
    Get an existing logger instance.

    Args:
        name: Logger name (default: "lakehouse")

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default logger instance
_default_logger: Optional[logging.Logger] = None


def get_default_logger() -> logging.Logger:
    """
    Get or create the default lakehouse logger.

    Returns:
        Default logger instance with INFO level and console output
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console_output: bool = True,
) -> None:
    """
    Configure the default lakehouse logger.

    Args:
        level: Logging level as string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Optional path to log file
        console_output: Whether to output logs to console (default: True)

    Example:
        >>> configure_logging(level="DEBUG", log_file=Path("lakehouse.log"))
    """
    global _default_logger
    _default_logger = setup_logger(
        name="lakehouse",
        level=level,
        log_file=log_file,
        console_output=console_output,
    )

