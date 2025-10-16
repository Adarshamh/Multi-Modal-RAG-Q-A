import os
import pickle
from typing import List
from sentence_transformers import SentenceTransformer
from backend.core.config import EMBED_DIR
from backend.core.logger import logger

os.makedirs(EMBED_DIR, exist_ok=True)

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info("Load pretrained SentenceTransformer: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

def embed_texts(texts: List[str]):
    model = get_model()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embs.tolist()

def save_embeddings(name: str, vectors, metadatas):
    path = os.path.join(EMBED_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"vectors": vectors, "metadatas": metadatas}, f)
    logger.info("Saved embeddings to %s", path)

def load_embeddings(name: str):
    path = os.path.join(EMBED_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        logger.info("Loaded embeddings from %s", path)
        return pickle.load(f)
