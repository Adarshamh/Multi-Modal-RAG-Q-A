# backend/core/audio_transcriber.py
import os
import tempfile
import uuid
from typing import Dict, Any, Optional
from .logger import logger

# whisper import (openai-whisper package)
import whisper

# Load model once at import time (adjust model size as needed)
# Warning: larger models use more RAM/VRAM and are slower.
try:
    MODEL_NAME = os.getenv("WHISPER_MODEL", "small")  # can be tiny, base, small, medium, large
    _whisper_model = whisper.load_model(MODEL_NAME)
    logger.info(f"Loaded Whisper model: {_whisper_model}")
except Exception as e:
    logger.exception("Failed to load Whisper model")
    _whisper_model = None


def _save_bytes_to_file(b: bytes, suffix: str = ".wav") -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(b)
    return tmp_path


def transcribe_audio_bytes(audio_bytes: bytes, language: Optional[str] = None, task: str = "transcribe") -> Dict[str, Any]:
    """
    Transcribe audio bytes using Whisper.
    Returns dict with 'text' and raw whisper result.
    """
    if _whisper_model is None:
        logger.error("Whisper model not loaded")
        return {"error": "Whisper model not available on server."}

    try:
        # write temp file
        tmpfile = _save_bytes_to_file(audio_bytes, suffix=".wav")
        # call whisper; set language if provided and task transcribe or translate
        options = {"task": task}
        if language:
            options["language"] = language
        result = _whisper_model.transcribe(tmpfile, **options)
        text = result.get("text", "").strip()
        # clean up
        try:
            os.remove(tmpfile)
        except Exception:
            pass
        return {"text": text, "whisper_result": result}
    except Exception as e:
        logger.exception("Whisper transcription failed")
        return {"error": str(e)}
