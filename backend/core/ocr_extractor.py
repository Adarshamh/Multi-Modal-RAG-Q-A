from PIL import Image
import pytesseract
from .logger import logger
from .config import TESSERACT_CMD

# Optionally set tesseract path if provided
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_text_from_image_bytes(image_bytes) -> str:
    try:
        img = Image.open(image_bytes)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logger.exception("OCR extraction failed")
        return ""
