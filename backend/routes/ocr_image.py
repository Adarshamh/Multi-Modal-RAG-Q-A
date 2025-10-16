from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.core.ocr_extractor import extract_text_from_image_upload
from backend.core.logger import logger

router = APIRouter()

@router.post("/extract-text-from-image")
async def extract_image(file: UploadFile = File(...)):
    try:
        text = extract_text_from_image_upload(file)
        if not text:
            return {"answer": "⚠️ No text extracted."}
        return {"answer": text}
    except Exception as e:
        logger.exception("ocr route error")
        raise HTTPException(status_code=500, detail=str(e))
