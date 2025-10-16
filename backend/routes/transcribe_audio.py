import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..core.text_extractor import transcribe_audio
from ..core.config import UPLOAD_DIR
from ..core.logger import logger

router = APIRouter()

@router.post("/transcribe-audio")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        text = transcribe_audio(file_path)
        return {"answer": text or "⚠️ No transcript returned."}
    except Exception as e:
        logger.exception("transcribe error")
        return JSONResponse(status_code=500, content={"error": str(e)})
