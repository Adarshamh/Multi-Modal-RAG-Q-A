from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import WebBaseLoader
from backend.core.rag_engine import retrieve
from backend.core.model_selector import select_model
from backend.core.logger import logger
import ollama, os

router = APIRouter()

class URLQuery(BaseModel):
    url: str
    question: str
    template: str = "qa"
    session_id: str = None

@router.post("/chat-with-url")
async def chat_with_url(data: URLQuery):
    try:
        os.environ["USER_AGENT"] = "Mozilla/5.0"
        loader = WebBaseLoader(data.url)
        docs = loader.load()
        content = "\n".join([d.page_content for d in docs])
        kb_ctx_docs = retrieve(data.question, k=3)
        kb_context = "\n\n".join([d["page_content"] for d in kb_ctx_docs]) if kb_ctx_docs else ""
        prompt = f"Context:\n{content}\n\nKB:\n{kb_context}\n\nQuestion: {data.question}"
        model = select_model("text")
        resp = ollama.chat(model=model, messages=[{"role":"user","content":prompt}])
        answer = resp.get("text") if isinstance(resp, dict) and resp.get("text") else str(resp)
        return {"answer": answer}
    except Exception as e:
        logger.exception("chat_with_url error")
        raise HTTPException(status_code=500, detail=str(e))
