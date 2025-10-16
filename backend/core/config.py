import os
from dotenv import load_dotenv

# Load .env in repo root (backend/.env or project .env)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")
load_dotenv(ENV_PATH)

# Directories
BASE_DIR = ROOT
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "..", "data"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "..", "data", "uploads"))
EMBED_DIR = os.getenv("EMBED_DIR", os.path.join(BASE_DIR, "..", "data", "embeddings"))
LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "..", "logs"))
LOG_PATH = os.getenv("LOG_PATH", os.path.join(LOG_DIR, "app.log"))
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "..", "data", "app.db"))

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "llama3:latest")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")
OLLAMA_LLAVA_MODEL = os.getenv("OLLAMA_LLAVA_MODEL", "llava-llama3")

# 👇 Added alias for backward compatibility
# (many routes expect `OLLAMA_MODEL` to point to the default text model)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", OLLAMA_TEXT_MODEL)

# Limits and chunking
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 200))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# External dependencies
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "")

# Allowed file types
ALLOWED_EXTENSIONS = {
    ".txt", ".pdf", ".docx", ".csv", ".png", ".jpg", ".jpeg",
    ".mp3", ".wav", ".mp4", ".mov", ".py", ".js", ".cs", ".md"
}

# Ensure directories exist
os.makedirs(os.path.join(BASE_DIR, "..", "data"), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
