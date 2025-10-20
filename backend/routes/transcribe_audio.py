# backend/routes/transcribe_audio.py
import os
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ..core.audio_transcriber import transcribe_audio_bytes
from ..core.logger import logger
from ..core.rag_engine import add_document_to_index  # keep existing add-to-kb logic if you want transcript saved
from typing import Optional

router = APIRouter()


@router.post("/transcribe-audio")
async def transcribe_audio(
    file: UploadFile = File(...),
    add_to_kb: Optional[str] = Form(None),  # "true"/"false"
    filename: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    """
    Single-shot transcription endpoint. Returns JSON with 'answer' containing transcription text.
    Optionally add transcript to KB by sending add_to_kb=true.
    """
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(status_code=400, content={"error": "Empty audio file."})

        result = transcribe_audio_bytes(contents, language=language)
        if "error" in result:
            return JSONResponse(status_code=500, content={"error": result["error"]})
        text = result.get("text", "")

        # Optionally add to KB
        if add_to_kb and add_to_kb.lower() in ("1", "true", "yes"):
            safe_name = filename or f"transcript_{os.path.basename(file.filename)}"
            try:
                add_document_to_index(safe_name, text)
                logger.info(f"Added transcript to KB: {safe_name}")
            except Exception:
                logger.exception("Failed to add transcript to KB")

        return JSONResponse(content={"answer": text, "meta": {"source": "whisper"}})
    except Exception as e:
        logger.exception("transcribe-audio error")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/transcribe-partial")
async def transcribe_partial(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """
    Accepts a short audio chunk (wav) and returns a partial transcription result.
    Designed for the frontend to send repeated short chunks to emulate live captions.
    """
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(status_code=400, content={"error": "Empty audio chunk."})
        result = transcribe_audio_bytes(contents, language=language)
        if "error" in result:
            return JSONResponse(status_code=500, content={"error": result["error"]})
        text = result.get("text", "")
        return JSONResponse(content={"partial": text})
    except Exception as e:
        logger.exception("transcribe-partial error")
        return JSONResponse(status_code=500, content={"error": str(e)})
