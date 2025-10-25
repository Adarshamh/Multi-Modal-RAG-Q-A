import logging
import sys
from .config import LOG_FILE

logger = logging.getLogger("mm_rag")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers if imported multiple times
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
