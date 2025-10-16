import logging
from .config import LOG_PATH, LOG_DIR
import os

# Ensure log dir
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger("mm_rag")
# also stream to console for dev
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
console.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console)
logger.info("Logger initialized, logs will be saved to %s", LOG_PATH)