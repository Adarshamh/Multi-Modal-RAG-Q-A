import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..core.text_extractor import extract_text_from_image
from ..core.config import UPLOAD_DIR
from ..core.logger import logger

router = APIRouter()

@router.post("/extract-text-from-image")
async def extract_image(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        text = extract_text_from_image(file_path)
        return {"answer": text or "⚠️ No text extracted."}
    except Exception as e:
        logger.exception("ocr error")
        return JSONResponse(status_code=500, content={"error": str(e)})
