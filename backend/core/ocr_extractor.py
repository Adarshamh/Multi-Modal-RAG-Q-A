from PIL import Image
import pytesseract
import io
import os
from .logger import logger
from .config import TESSERACT_CMD

# 🔧 Auto-resolve tesseract path on Windows if not found
if not TESSERACT_CMD or TESSERACT_CMD == "tesseract":
    default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_path):
        pytesseract.pytesseract.tesseract_cmd = default_path
    else:
        pytesseract.pytesseract.tesseract_cmd = "tesseract"
else:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def extract_text_from_image_bytes(image_bytes) -> str:
    try:
        img = Image.open(image_bytes if hasattr(image_bytes, 'read') else io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        logger.exception("OCR extraction failed")
        return ""
