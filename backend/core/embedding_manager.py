import os
import pickle
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from ..core.config import EMBED_DIR
from ..core.logger import logger

os.makedirs(EMBED_DIR, exist_ok=True)

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[SentenceTransformer] = None

def get_model():
    global _model
    if _model is None:
        logger.info(f"Load pretrained SentenceTransformer: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embs.tolist()

def save_embeddings(name: str, vectors, metadatas):
    path = os.path.join(EMBED_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"vectors": vectors, "metadatas": metadatas}, f)
    logger.info(f"Saved embeddings to {path}")

def load_embeddings(name: str):
    path = os.path.join(EMBED_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded embeddings from {path}")
    return data
