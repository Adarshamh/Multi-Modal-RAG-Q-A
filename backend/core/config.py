import os
from dotenv import load_dotenv

# load .env from backend dir (if exists)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    load_dotenv()  # fallback to environment

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "..", "data", "uploads"))
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", os.path.join(BASE_DIR, "..", "data", "embeddings"))
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "embed_index.faiss")
IMAGE_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "image_index.faiss")
CHUNK_STORE_FILE = os.path.join(EMBEDDINGS_DIR, "chunk_store.pkl")

DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "..", "data", "app.db"))

LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "..", "logs"))
LOG_FILE = os.getenv("LOG_FILE", os.path.join(LOG_DIR, "app.log"))

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "llama3:latest")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# limits and chunking
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 200))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# OCR / audio
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "")

ALLOWED_EXTENSIONS = set([
    ".txt", ".pdf", ".docx", ".csv", ".png", ".jpg", ".jpeg", ".mp3", ".wav", ".mp4", ".mov",
    ".py", ".js", ".md"
])

# Ensure folders exist
os.makedirs(os.path.join(BASE_DIR, "..", "data"), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
