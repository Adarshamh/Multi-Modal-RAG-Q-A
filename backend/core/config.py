# backend/core/config.py
import os
from dotenv import load_dotenv

# Load environment variables
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    # also try project root .env
    load_dotenv()

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(DATA_DIR, "uploads"))
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", os.path.join(DATA_DIR, "embeddings"))
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "embed_index.faiss")
CHUNK_STORE_FILE = os.path.join(EMBEDDINGS_DIR, "chunk_store.pkl")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "llama3:latest")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Limits & chunking
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 200))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# OCR / audio
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "")

# Allowed file types
ALLOWED_EXTENSIONS = {
    ".txt", ".pdf", ".docx", ".csv", ".png", ".jpg", ".jpeg",
    ".mp3", ".wav", ".mp4", ".mov", ".py", ".js", ".md"
}

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Default DB path (if required)
DB_PATH = os.path.join(DATA_DIR, "rag_metadata.db")
