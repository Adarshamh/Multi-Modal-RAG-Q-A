"""
High-level RAG API: add_document_to_index, retrieve
Uses EmbeddingManager (dense + sparse hybrid) for retrieval.
"""
from .embedding_manager import get_manager
from .logger import logger

def add_document_to_index(name: str, text: str, meta: dict = None):
    m = get_manager()
    added = m.add_documents(name, text, meta=meta)
    logger.info("Added %d chunks to index for %s", added, name)
    return added

def add_image_to_index(name: str, pil_image, meta: dict = None):
    m = get_manager()
    added = m.add_image(name, pil_image, meta=meta)
    logger.info("Added image embedding for %s", name)
    return added

def retrieve(query: str, k: int = 3, alpha: float = 0.6):
    m = get_manager()
    results = m.hybrid_search(query, k=k, alpha=alpha)
    return results
