import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / f"{datetime.now():%m_%d_%Y_%H_%M_%S}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024
BACKUP_COUNT = 3


def configure_logger(name: str = "doc_compare") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:  # duplicate handlers se bachne ke liye
        return log

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)
    log.propagate = False

    fmt = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)

    log.addHandler(file_handler)
    log.addHandler(console)
    return log


logger = configure_logger()