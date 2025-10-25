from PIL import Image
import pytesseract
from .logger import logger
from .config import TESSERACT_CMD
from io import BytesIO

# set tesseract path if provided
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_text_from_image_bytes(image_bytes) -> str:
    try:
        if isinstance(image_bytes, (bytes, bytearray)):
            img = Image.open(BytesIO(image_bytes))
        else:
            img = Image.open(image_bytes)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logger.exception("OCR extraction failed")
        return ""
