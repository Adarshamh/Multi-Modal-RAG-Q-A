import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.core.text_extractor import transcribe_audio
from backend.core.config import UPLOAD_DIR
from backend.core.logger import logger

router = APIRouter()

@router.post("/transcribe-audio")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        text = transcribe_audio(file_path)
        return {"answer": text or "⚠️ No transcript returned."}
    except Exception as e:
        logger.exception("transcribe_audio error")
        raise HTTPException(status_code=500, detail=str(e))
