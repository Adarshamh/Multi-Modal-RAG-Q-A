from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..core.ocr_extractor import extract_text_from_image_bytes
from ..core.logger import logger

router = APIRouter()

@router.post("/extract-text-from-image")
async def extract_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = extract_text_from_image_bytes(contents)
        return {"answer": text}
    except Exception as e:
        logger.exception("OCR route error")
        return JSONResponse(status_code=500, content={"error": str(e)})
