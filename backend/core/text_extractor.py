import os
from docx import Document
import pandas as pd
import PyPDF2
from PIL import Image
import pytesseract
from moviepy.editor import AudioFileClip
# whisper import if available - fallback to external call
try:
    from whisper import load_model
    _HAS_WHISPER = True
except Exception:
    _HAS_WHISPER = False

from ..core.config import TESSERACT_CMD, FFMPEG_PATH, UPLOAD_DIR
from ..core.logger import logger

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD or "tesseract"
if FFMPEG_PATH:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH

_whisper_model = None

def _ensure_whisper():
    global _whisper_model
    if not _HAS_WHISPER:
        return None
    if _whisper_model is None:
        _whisper_model = load_model("base")
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
            logger.exception("PDF extract error")
            return ""
    if ext in [".py", ".js", ".cs", ".md", ".json"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    # fallback: none
    return ""

def extract_text_from_image(file_path: str) -> str:
    try:
        img = Image.open(file_path)
        txt = pytesseract.image_to_string(img)
        return txt.strip()
    except Exception:
        logger.exception("OCR error")
        return ""

def transcribe_audio(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".mp4", ".mov", ".mkv", ".avi"]:
        clip = AudioFileClip(file_path)
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"
        clip.audio.write_audiofile(wav_path, logger=None)
        file_path = wav_path
    model = _ensure_whisper()
    if model:
        result = model.transcribe(file_path)
        return result.get("text", "")
    else:
        # whisper not installed - can't transcribe locally
        logger.warning("Whisper not available; returning empty transcript")
        return ""
