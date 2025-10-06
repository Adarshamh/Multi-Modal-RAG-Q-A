from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document as LDoc
import os

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = None
kb_docs = []

def add_to_kb(file_content: str):
    global vector_store
    doc = LDoc(page_content=file_content)
    kb_docs.append(doc)
    vector_store = FAISS.from_documents(kb_docs, embeddings)

def retrieve_from_kb(question: str, top_k: int = 5):
    if not vector_store:
        return ""
    docs = vector_store.similarity_search(question, k=top_k)
    return "\n".join([d.page_content for d in docs])
