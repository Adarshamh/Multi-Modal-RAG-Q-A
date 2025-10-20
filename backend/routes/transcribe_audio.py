import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..core.logger import logger

router = APIRouter()

@router.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Tries whisper (if installed) then falls back to SpeechRecognition with pocketsphinx (if installed).
    Save audio to temp file and run.
    """
    try:
        contents = await file.read()
        from tempfile import NamedTemporaryFile
        tf = NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        tf.write(contents)
        tf.flush()
        tf.close()
        tmp_path = tf.name

        # try whisper first (faster/better if GPU available)
        try:
            import whisper
            model = whisper.load_model("small")
            res = model.transcribe(tmp_path)
            text = res.get("text", "")
            return {"ok": True, "answer": text}
        except Exception:
            logger.info("Whisper not available or failed, trying SpeechRecognition as fallback")

        # fallback: SpeechRecognition
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(tmp_path) as source:
                audio = r.record(source)
            text = r.recognize_sphinx(audio)
            return {"ok": True, "answer": text}
        except Exception as e:
            logger.exception("SpeechRecognition fallback failed")
            return JSONResponse(status_code=500, content={"ok": False, "error": "Transcription failed", "detail": str(e)})
    except Exception as e:
        logger.exception("transcribe_audio error")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
