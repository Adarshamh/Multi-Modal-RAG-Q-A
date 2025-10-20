from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..core.ocr_extractor import extract_text_from_image_bytes
from ..core.logger import logger

router = APIRouter()

@router.post("/extract-text-from-image")
async def extract_text_from_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        from io import BytesIO
        bio = BytesIO(contents)
        text = extract_text_from_image_bytes(bio)
        return {"ok": True, "answer": text}
    except Exception as e:
        logger.exception("OCR endpoint error")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
