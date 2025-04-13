import logging
import logging.handlers
from pathlib import Path


def setup_logging(config):
    """Set up centralized logging for the application"""
    # Get logging config values
    log_level = config.get("application", {}).get("log_level", "INFO")
    log_file = config.get("application", {}).get("log_file", "application.log")
    log_format = config.get("application", {}).get(
        "log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    max_size = config.get("application", {}).get(
        "log_max_size", 5 * 1024 * 1024
    )  # 5 MB default
    backup_count = config.get("application", {}).get("log_backup_count", 3)

    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Create and add file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
