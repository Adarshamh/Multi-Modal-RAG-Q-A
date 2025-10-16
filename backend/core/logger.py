import logging
from logging.handlers import RotatingFileHandler
from .config import LOG_PATH

logger = logging.getLogger("mm_rag")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    try:
        fh = RotatingFileHandler(LOG_PATH, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    except Exception:
        logger.warning("Could not create file log handler; continuing with console logging.")
        