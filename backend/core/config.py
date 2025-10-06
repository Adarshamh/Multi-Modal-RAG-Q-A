import os
from dotenv import load_dotenv

load_dotenv()

# Paths
UPLOAD_DIR = os.path.join(os.getcwd(), "data", "uploads")
EMBEDDING_DIR = os.path.join(os.getcwd(), "data", "embeddings")
CACHE_DIR = os.path.join(os.getcwd(), "data", "cache")
LOG_FILE = os.path.join(os.getcwd(), "logs", "app.log")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Config
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 10))
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
VECTOR_DB = os.getenv("VECTOR_DB", "chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
