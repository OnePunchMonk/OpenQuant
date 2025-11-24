import logging
import sys
from pathlib import Path
from typing import List
from src.utils.config import config

LOG_FILE_PATH = Path("logs") / "app.log"


def setup_logging():
    """Configure root logger with console and rotating file handler."""
    log_level = config.get("log_level", "INFO").upper()
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    LOG_FILE_PATH.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info(f"Logging initialized with level {log_level}")


def tail_log(lines: int = 50) -> List[str]:
    """Return the last N lines of the application log file.

    Args:
        lines: Number of lines from the end of the log to return.
    Returns:
        List[str]: The log lines (newest last). If file absent, returns empty list.
    """
    if not LOG_FILE_PATH.exists():
        return []
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.readlines()
        return content[-lines:]
    except Exception:
        return []