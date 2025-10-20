from .embedding_manager import get_manager
from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .logger import logger

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.replace("\r\n", "\n")
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        # move with overlap
        start = end - overlap if end - overlap > start else end
        if start >= len(text):
            break
    return chunks

def add_document_to_index(doc_name, text):
    manager = get_manager()
    chunks = chunk_text(text)
    metadata = [{"source": doc_name, "chunk_idx": i} for i in range(len(chunks))]
    added = manager.add_chunks(chunks, metadata)
    logger.info("Added %d chunks to index for %s", added, doc_name)
    return added

def retrieve(query, k=3):
    manager = get_manager()
    results = manager.search(query, k)
    return results
