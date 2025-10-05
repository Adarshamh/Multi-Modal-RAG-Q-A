from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

embedding_model = HuggingFaceEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def create_knowledge_base(file_content):
    chunks = text_splitter.split_text(file_content)
    vector_store = FAISS.from_texts(chunks, embedding_model)
    return vector_store

def retrieve_from_kb(vector_store, query, top_k=5):
    docs = vector_store.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs])
