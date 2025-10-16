from fastapi import UploadFile
from io import BytesIO
from PIL import Image
import pytesseract
from backend.core.logger import logger

def extract_text_from_image_upload(file: UploadFile) -> str:
    try:
        logger.info("OCR: processing file %s", file.filename)
        data = file.file.read()
        img = Image.open(BytesIO(data))
        text = pytesseract.image_to_string(img)
        txt = text.strip()
        if not txt:
            logger.info("OCR: no text found in %s", file.filename)
            return ""
        return txt
    except Exception as e:
        logger.exception("OCR failed: %s", e)
        return ""
