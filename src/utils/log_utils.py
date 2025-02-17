"""
Logging configuration for the project.
Implements structured logging with rotation and different log levels.
"""

import logging
import logging.handlers
from pathlib import Path
import sys
from datetime import datetime
from functools import wraps
import time
from typing import Callable, Any

from src.config.paths import LOGS_DIR

# Configure logging formats
CONSOLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FILE_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def setup_logging(name: str) -> logging.Logger:
    """
    Set up logging with both file and console handlers.

    Args:
        name: Logger name (typically __name__ of the calling module)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    console_handler.setLevel(logging.INFO)

    # File Handler with rotation
    today = datetime.now().strftime("%Y%m%d")
    log_file = LOGS_DIR / f"{name.split('.')[-1]}_{today}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}"
            )
            raise

    return wrapper
