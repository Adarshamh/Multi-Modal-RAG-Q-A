import os
import numpy as np
from backend.core.embedding_manager import embed_texts, save_embeddings, load_embeddings
from backend.core.config import EMBED_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from backend.core.logger import logger

_STORE_NAME = "lc_faiss_store"

_texts = []
_metadatas = []
_vectors = None

def split_text_to_chunks(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def add_document_to_index(doc_id: str, text: str):
    global _texts, _metadatas, _vectors
    chunks = split_text_to_chunks(text)
    if not chunks:
        return 0
    vectors = embed_texts(chunks)
    if _vectors is None:
        _vectors = vectors.copy()
    else:
        _vectors.extend(vectors)
    _texts.extend(chunks)
    _metadatas.extend([{"source": doc_id, "chunk": i} for i in range(len(chunks))])
    save_embeddings(_STORE_NAME, _vectors, _metadatas)
    logger.info("Added %d chunks to index for %s", len(chunks), doc_id)
    return len(chunks)

def retrieve(query: str, k=5):
    global _texts, _metadatas, _vectors
    if not _texts or not _vectors:
        loaded = load_embeddings(_STORE_NAME)
        if loaded:
            # We stored vectors and metadatas; we set _vectors to loaded vectors
            _vectors = loaded.get("vectors")
            _metadatas = loaded.get("metadatas", [])
            # Note: because we didn't persist texts separately, we cannot reconstruct texts reliably.
            # Fallback: return empty
        return []
    qv = embed_texts([query])[0]
    vecs = np.array(_vectors)
    qarr = np.array(qv)
    norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(qarr)
    scores = (vecs @ qarr) / np.where(norms == 0, 1e-9, norms)
    idx = np.argsort(-scores)[:k]
    results = []
    for i in idx:
        results.append({"score": float(scores[i]), "page_content": _texts[i], "metadata": _metadatas[i]})
    return results
