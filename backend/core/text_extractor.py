import os
from docx import Document
import pandas as pd
import PyPDF2
from PIL import Image
import pytesseract
from moviepy.editor import AudioFileClip
from whisper import load_model
from backend.core.config import TESSERACT_CMD, FFMPEG_PATH, UPLOAD_DIR
from backend.core.logger import logger

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
if FFMPEG_PATH:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH

_whisper_model = None

def _ensure_whisper():
    global _whisper_model
    if _whisper_model is None:
        try:
            _whisper_model = load_model("base")
            logger.info("Whisper model loaded")
        except Exception as e:
            logger.exception("Failed to load Whisper: %s", e)
            _whisper_model = None
    return _whisper_model

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    if ext == ".csv":
        try:
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
        except Exception:
            return ""
    if ext == ".pdf":
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = []
                for p in reader.pages:
                    txt = p.extract_text()
                    if txt:
                        pages.append(txt)
                return "\n".join(pages)
        except Exception:
            return ""
    if ext in [".py", ".js", ".cs", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

def extract_text_from_image(file_path: str) -> str:
    try:
        img = Image.open(file_path)
        txt = pytesseract.image_to_string(img)
        return txt.strip()
    except Exception as e:
        logger.exception("extract_text_from_image failed: %s", e)
        return ""

def transcribe_audio(file_path: str) -> str:
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".mp4", ".mov", ".mkv", ".avi"]:
            clip = AudioFileClip(file_path)
            wav_path = file_path.rsplit(".", 1)[0] + ".wav"
            clip.audio.write_audiofile(wav_path, logger=None)
            file_path = wav_path
        model = _ensure_whisper()
        if not model:
            return ""
        result = model.transcribe(file_path)
        return result.get("text", "")
    except Exception as e:
        logger.exception("transcribe_audio failed: %s", e)
        return ""
