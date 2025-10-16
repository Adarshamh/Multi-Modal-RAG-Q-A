import os
import heapq
from typing import List, Tuple, Optional
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LDoc
from ..core.config import EMBED_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from ..core.embedding_manager import embed_texts, save_embeddings, load_embeddings
from ..core.logger import logger

_index: Optional[FAISS] = None
_texts: List[str] = []
_metadatas: List[dict] = []

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

def _ensure_index():
    global _index
    if _index is None:
        loaded = load_embeddings("lc_faiss_store")
        if loaded:
            # try to reconstruct empty FAISS object using LangChain's persistence is complex.
            # We'll create index from texts if available
            vectors = loaded.get("vectors", None)
            metadatas = loaded.get("metadatas", None)
            if vectors and metadatas:
                try:
                    docs = [LDoc(page_content=metadatas[i].get("text","") if isinstance(metadatas[i], dict) else "", metadata=metadatas[i]) for i in range(len(metadatas))]
                    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                    _index = FAISS.from_documents(docs, embeddings)  # best-effort
                    logger.info("Rebuilt FAISS index from pickled store (best-effort)")
                except Exception:
                    logger.warning("Could not rebuild FAISS from pickle; starting fresh.")
        else:
            # no persisted store - will be created when adding docs
            pass
    return _index

def add_document_to_index(doc_id: str, text: str):
    global _index, _texts, _metadatas
    chunks = split_text_to_chunks(text)
    if not chunks:
        return 0
    # embed in batch
    vectors = embed_texts(chunks)
    # create LangChain documents
    docs = [LDoc(page_content=chunks[i], metadata={"source": doc_id, "chunk": i, "text": chunks[i]}) for i in range(len(chunks))]
    # create FAISS index if not present
    if _index is None:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        _index = FAISS.from_documents(docs, embeddings)
    else:
        _index.add_documents(docs)
    # persist - simplified: save lists
    _texts.extend(chunks)
    _metadatas.extend([{"source": doc_id, "chunk": i, "text": chunks[i]} for i in range(len(chunks))])
    save_embeddings("lc_faiss_store", _texts, _metadatas)
    logger.info(f"Added {len(chunks)} chunks to index for {doc_id}")
    return len(chunks)

def retrieve(query: str, k=5):
    global _index
    if _index is None:
        _index = _ensure_index()
        if _index is None:
            return []
    try:
        docs = _index.similarity_search(query, k=k)
        return docs
    except Exception as e:
        logger.exception("retrieve failed")
        return []
