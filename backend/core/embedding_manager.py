import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from .config import EMBEDDINGS_DIR, FAISS_INDEX_FILE
from .logger import logger

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logger.info("Load pretrained SentenceTransformer: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self._load_index()
        # chunk store maps vector index id -> chunk dict
        self.chunk_store = self._load_chunk_store()

    def _load_index(self):
        if os.path.exists(FAISS_INDEX_FILE):
            try:
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logger.info("Loaded persisted FAISS index from %s", FAISS_INDEX_FILE)
            except Exception as e:
                logger.exception("Failed to read FAISS index, rebuilding new index")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        # using IndexFlatL2 for simplicity
        self.index = faiss.IndexFlatL2(self.dim)
        logger.info("Created a new FAISS index (IndexFlatL2) with dim=%d", self.dim)

    def _load_chunk_store(self):
        store_path = os.path.join(EMBEDDINGS_DIR, "chunk_store.pkl")
        if os.path.exists(store_path):
            try:
                with open(store_path, "rb") as f:
                    store = pickle.load(f)
                logger.info("Loaded chunk_store with %d entries", len(store))
                return store
            except Exception:
                logger.exception("Failed to load chunk_store, init empty")
        return []

    def _save_chunk_store(self):
        store_path = os.path.join(EMBEDDINGS_DIR, "chunk_store.pkl")
        with open(store_path, "wb") as f:
            pickle.dump(self.chunk_store, f)
        logger.info("Saved chunk_store (%d chunks) to %s", len(self.chunk_store), store_path)

    def save_index(self):
        try:
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logger.info("Saved FAISS index to %s", FAISS_INDEX_FILE)
        except Exception:
            logger.exception("Error saving FAISS index")

    def embed_texts(self, texts):
        # returns numpy array (n, dim)
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs

    def add_chunks(self, chunks, metadata=None):
        """
        chunks: list of text chunks
        metadata: optional metadata with each chunk
        returns number added
        """
        if not chunks:
            return 0
        embs = self.embed_texts(chunks)
        self.index.add(embs)
        base_id = len(self.chunk_store)
        for i, c in enumerate(chunks):
            entry = {
                "id": base_id + i,
                "text": c,
                "meta": metadata[i] if metadata and i < len(metadata) else {}
            }
            self.chunk_store.append(entry)
        self._save_chunk_store()
        self.save_index()
        logger.info("Added %d chunks to index", len(chunks))
        return len(chunks)

    def search(self, query, k=3):
        q_emb = self.embed_texts([query])
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(q_emb, k)
        results = []
        for idx in I[0]:
            if idx < len(self.chunk_store):
                results.append(self.chunk_store[idx])
            else:
                # safety
                results.append({"id": idx, "text": "[missing chunk]"})
        return results

# Singleton manager
_manager = None

def get_manager():
    global _manager
    if _manager is None:
        _manager = EmbeddingManager()
    return _manager
