from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
import whisper
from moviepy.editor import AudioFileClip
from core.config import UPLOAD_DIR
import os

router = APIRouter()
whisper_model = whisper.load_model("base")

@router.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile):
    try:
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in [".mp4", ".mov"]:
            clip = AudioFileClip(file_path)
            audio_path = file_path.replace(ext, ".wav")
            clip.audio.write_audiofile(audio_path)
            file_path = audio_path
        result = whisper_model.transcribe(file_path)
        return {"answer": result.get("text", "⚠️ No transcript returned.")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
