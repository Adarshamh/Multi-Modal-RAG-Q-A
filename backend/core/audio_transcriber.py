import io
import os
import tempfile
import base64
from ..core.logger import logger
from ..core.config import FFMPEG_PATH
import whisper
from pydub import AudioSegment

# Ensure ffmpeg path is available to subprocesses (try to help Whisper)
if FFMPEG_PATH:
    ff_dir = os.path.dirname(FFMPEG_PATH)
    os.environ["PATH"] = ff_dir + os.pathsep + os.environ.get("PATH", "")

# load whisper model lazily
_whisper_model = None
def get_whisper_model(size="small"):
    global _whisper_model
    if _whisper_model is None:
        try:
            _whisper_model = whisper.load_model(size)
            logger.info("Loaded Whisper model: %s", size)
        except Exception as e:
            logger.exception("Failed to load Whisper model")
            raise
    return _whisper_model

def transcribe_audio_bytes(file_bytes, model_size="small", language=None, task="transcribe"):
    """
    Single-shot transcription: accepts raw audio bytes (wav, mp3, m4a).
    """
    try:
        # write tmp file with proper extension by trying to detect format via pydub
        tmpfd, tmpname = tempfile.mkstemp(suffix=".wav")
        os.close(tmpfd)
        # use pydub to convert bytes to wav
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        audio.export(tmpname, format="wav")
        model = get_whisper_model(model_size)
        options = {"task": task}
        if language:
            options["language"] = language
        res = model.transcribe(tmpname, **options)
        text = res.get("text", "")
        # cleanup
        try:
            os.remove(tmpname)
        except:
            pass
        return text
    except Exception as e:
        logger.exception("Whisper transcription failed")
        return ""

# Simple chunked appending transcription helper for "live" mode
# Server-side we append received audio chunks to a temp file and when final==true we transcribe
_live_buffers = {}  # session_id -> temp file path

def append_chunk_and_maybe_transcribe(session_id: str, chunk_b64: str, final: bool = False, model_size="small"):
    """
    Accept base64-encoded audio chunk, append to a per-session buffer and, if final True, transcribe.
    Returns None if not final, otherwise returns transcript string.
    """
    try:
        b = base64.b64decode(chunk_b64)
        if session_id not in _live_buffers:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            _live_buffers[session_id] = path
        else:
            path = _live_buffers[session_id]

        # Append chunk - pydub helps unify
        seg = AudioSegment.from_file(io.BytesIO(b))
        if os.path.exists(path) and os.path.getsize(path) > 0:
            existing = AudioSegment.from_file(path)
            combined = existing + seg
            combined.export(path, format="wav")
        else:
            seg.export(path, format="wav")

        if final:
            model = get_whisper_model(model_size)
            res = model.transcribe(path)
            text = res.get("text", "")
            # cleanup
            try:
                os.remove(path)
            except:
                pass
            _live_buffers.pop(session_id, None)
            return text
        else:
            return None
    except Exception:
        logger.exception("Chunk append/transcribe failed")
        return "[ERROR]"
