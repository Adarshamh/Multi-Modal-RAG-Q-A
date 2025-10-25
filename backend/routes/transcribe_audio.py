import json
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ..core.audio_transcriber import transcribe_audio_bytes, append_chunk_and_maybe_transcribe
from ..core.logger import logger

router = APIRouter()

@router.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...), model: str = Form("small")):
    try:
        contents = await file.read()
        text = transcribe_audio_bytes(contents, model_size=model)
        return {"answer": text}
    except Exception as e:
        logger.exception("transcribe_audio error")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/transcribe-stream")
async def transcribe_stream(session_id: str = Form(...), chunk_b64: str = Form(...), final: bool = Form(False), model: str = Form("small")):
    """
    Accept base64-encoded chunk and append server-side.
    When final=True, transcribe full recording and return the transcript.
    This simple API enables live capture: the frontend can post many small chunks and set final when done.
    """
    try:
        text_or_none = append_chunk_and_maybe_transcribe(session_id, chunk_b64, final=final, model_size=model)
        if final:
            return {"answer": text_or_none}
        else:
            return {"ok": True}
    except Exception as e:
        logger.exception("transcribe-stream error")
        return JSONResponse(status_code=500, content={"error": str(e)})
