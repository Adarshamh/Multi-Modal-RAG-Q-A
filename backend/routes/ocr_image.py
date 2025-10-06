from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
from core.config import TESSERACT_CMD, UPLOAD_DIR
import os

router = APIRouter()
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

@router.post("/extract-text-from-image")
async def extract_image(file: UploadFile):
    try:
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image).strip()
        return {"answer": text or "⚠️ No text extracted."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
