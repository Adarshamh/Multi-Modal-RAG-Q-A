"""
EmbeddingManager: handles text & image embeddings, persistence and hybrid retrieval.
Uses:
 - SentenceTransformer for text embeddings
 - CLIP (transformers) for image embeddings (vision-language)
 - FAISS for dense vector index (text and images stored separately)
 - BM25 (rank_bm25) for sparse retrieval
 - simple persistent chunk_store (pickle)
"""
import os
import pickle
import numpy as np
from threading import Lock
from sentence_transformers import SentenceTransformer
import faiss
from cachetools import LRUCache, cached
from rank_bm25 import BM25Okapi
from transformers import CLIPModel, CLIPProcessor
from ..core.config import EMBEDDINGS_DIR, FAISS_INDEX_FILE, IMAGE_INDEX_FILE, CHUNK_STORE_FILE, CHUNK_SIZE, CHUNK_OVERLAP
from ..core.logger import logger

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

_lock = Lock()

class EmbeddingManager:
    def __init__(self):
        # Text model
        self._text_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded text embedder: all-MiniLM-L6-v2")
        # CLIP for image embeddings
        try:
            self._clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("Loaded CLIP vision model")
        except Exception as e:
            self._clip = None
            self._clip_processor = None
            logger.warning("CLIP not loaded: %s", e)

        # In-memory chunk store: list of dicts {id, source, text, meta}
        self.chunk_store = self._load_chunk_store()
        # Build or load faiss index for text embeddings
        self.text_index, self.text_id_map = self._load_faiss(FAISS_INDEX_FILE)
        # Image index
        self.image_index, self.image_id_map = self._load_faiss(IMAGE_INDEX_FILE)
        # BM25 for sparse search
        self.bm25 = None
        self._build_bm25()
        # small LRU cache for embeddings
        self.emb_cache = LRUCache(1024)

    def _load_chunk_store(self):
        if os.path.exists(CHUNK_STORE_FILE):
            try:
                with open(CHUNK_STORE_FILE, "rb") as f:
                    store = pickle.load(f)
                logger.info("Loaded chunk_store with %d entries", len(store))
                return store
            except Exception:
                logger.exception("Failed loading chunk store")
        return []

    def _save_chunk_store(self):
        with open(CHUNK_STORE_FILE, "wb") as f:
            pickle.dump(self.chunk_store, f)
        logger.info("Saved chunk_store (%d chunks) to %s", len(self.chunk_store), CHUNK_STORE_FILE)

    def _load_faiss(self, path):
        if os.path.exists(path):
            try:
                idx = faiss.read_index(path)
                # load id map
                id_map = {}
                # we keep a simple list mapping: faiss index id -> store index
                logger.info("Loaded persisted FAISS index from %s", path)
                return idx, id_map
            except Exception:
                logger.exception("Failed to load faiss index %s", path)
        # empty index will be created on first add (d dimension known at runtime)
        return None, {}

    def _save_faiss(self, index, path):
        try:
            faiss.write_index(index, path)
            logger.info("Saved FAISS index to %s", path)
        except Exception:
            logger.exception("Failed to save FAISS index to %s", path)

    def _build_bm25(self):
        texts = [c.get("text", "") for c in self.chunk_store]
        if texts:
            tokenized = [t.split() for t in texts]
            self.bm25 = BM25Okapi(tokenized)
            logger.info("Built BM25 over %d documents", len(texts))
        else:
            self.bm25 = None

    def _ensure_text_index(self, dim):
        if self.text_index is None:
            # Use IndexFlatIP with normalizing to do cosine via inner product
            self.text_index = faiss.IndexFlatL2(dim)
            logger.info("Created new FAISS text index (dim=%d)", dim)

    def _ensure_image_index(self, dim):
        if self.image_index is None:
            self.image_index = faiss.IndexFlatL2(dim)
            logger.info("Created new FAISS image index (dim=%d)", dim)

    def embed_texts(self, texts):
        # caching by tuple of texts
        key = ("txt", tuple(texts))
        if key in self.emb_cache:
            return self.emb_cache[key]
        emb = self._text_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.emb_cache[key] = emb
        return emb

    def embed_text(self, text):
        return self.embed_texts([text])[0]

    def embed_image_bytes(self, image_pil):
        if not self._clip or not self._clip_processor:
            raise RuntimeError("CLIP model not available")
        inputs = self._clip_processor(images=image_pil, return_tensors="pt")
        with _lock:
            out = self._clip.get_image_features(**inputs)
        emb = out.detach().cpu().numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    def add_documents(self, source_name, text, meta=None):
        """
        Chunk text and add to chunk_store and text FAISS index
        returns number of chunks added
        """
        # chunk the text
        chunks = []
        i = 0
        L = len(text)
        while i < L:
            chunk = text[i:i + CHUNK_SIZE]
            chunks.append(chunk)
            i += CHUNK_SIZE - CHUNK_OVERLAP
        logger.info("Split %s into %d chunks", source_name, len(chunks))

        # embed all chunks
        embeddings = self.embed_texts(chunks)
        dim = embeddings.shape[1]
        self._ensure_text_index(dim)

        # add to faiss index and chunk_store
        for idx, chunk in enumerate(chunks):
            store_id = len(self.chunk_store)
            rec = {"id": store_id, "source": source_name, "text": chunk, "meta": meta or {}}
            self.chunk_store.append(rec)
            # add vector
            try:
                self.text_index.add(np.expand_dims(embeddings[idx], axis=0))
            except Exception as e:
                logger.exception("Failed to add vector: %s", e)
        # persist
        self._save_chunk_store()
        self._save_faiss(self.text_index, FAISS_INDEX_FILE)
        self._build_bm25()
        return len(chunks)

    def add_image(self, source_name, pil_image, meta=None):
        emb = self.embed_image_bytes(pil_image)
        dim = emb.shape[1] if emb.ndim > 1 else emb.shape[0]
        self._ensure_image_index(dim)
        try:
            self.image_index.add(emb)
            # store metadata as a chunk in chunk_store for retrieval linking
            store_id = len(self.chunk_store)
            rec = {"id": store_id, "source": source_name, "text": f"[IMAGE:{source_name}]", "meta": meta or {}, "is_image": True}
            self.chunk_store.append(rec)
            self._save_chunk_store()
            self._save_faiss(self.image_index, IMAGE_INDEX_FILE)
            self._build_bm25()
            return 1
        except Exception:
            logger.exception("Failed to add image embedding")
            return 0

    def search_dense(self, query, k=5):
        qv = self.embed_text(query).reshape(1, -1)
        if self.text_index is None or self.text_index.ntotal == 0:
            return []
        D, I = self.text_index.search(qv, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.chunk_store):
                results.append({"score": float(score), "chunk": self.chunk_store[idx]})
        return results

    def search_image_by_text(self, query, k=5):
        # embed query and search image index (cross-modal search)
        qv = self.embed_text(query).reshape(1, -1)
        if self.image_index is None or self.image_index.ntotal == 0:
            return []
        D, I = self.image_index.search(qv, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.chunk_store):
                results.append({"score": float(score), "chunk": self.chunk_store[idx]})
        return results

    def search_sparse(self, query, k=5):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_idx:
            if idx < len(self.chunk_store):
                results.append({"score": float(scores[idx]), "chunk": self.chunk_store[idx]})
        return results

    def hybrid_search(self, query, k=5, alpha=0.6):
        """
        Hybrid dense + sparse fusion:
        alpha * dense_score + (1-alpha) * sparse_score (normalized).
        """
        dense = self.search_dense(query, k=k * 2)
        sparse = self.search_sparse(query, k=k * 2)
        fused = {}
        # normalize dense scores (faiss L2 distances), convert to pseudo-similarity
        if dense:
            ds = np.array([1.0 / (1.0 + d["score"]) for d in dense])
            if ds.max() > 0:
                ds = ds / ds.max()
        else:
            ds = np.array([])

        sp_scores = []
        if sparse:
            sp_scores = np.array([s["score"] for s in sparse])
            if sp_scores.max() > 0:
                sp_scores = sp_scores / sp_scores.max()
        # combine
        for i, d in enumerate(dense):
            idx = d["chunk"]["id"]
            fused[idx] = fused.get(idx, 0.0) + alpha * (ds[i] if len(ds)>i else 0.0)
        for i, s in enumerate(sparse):
            idx = s["chunk"]["id"]
            fused[idx] = fused.get(idx, 0.0) + (1 - alpha) * (sp_scores[i] if len(sp_scores)>i else 0.0)
        # sort
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for idx, score in ranked:
            if idx < len(self.chunk_store):
                rec = self.chunk_store[idx].copy()
                rec["_score"] = float(score)
                results.append(rec)
        return results

# Singleton manager
_manager = None
def get_manager():
    global _manager
    if _manager is None:
        _manager = EmbeddingManager()
    return _manager
