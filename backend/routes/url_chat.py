from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import WebBaseLoader
from ..core.rag_engine import retrieve
from ..core.config import OLLAMA_HOST, OLLAMA_PORT, OLLAMA_TEXT_MODEL
from ..core.logger import logger
import os, requests

router = APIRouter()

class URLQuery(BaseModel):
    url: str
    question: str
    template: str = "qa"
    session_id: str = None

@router.post("/chat-with-url")
async def chat_with_url(data: URLQuery):
    try:
        # fetch page
        os.environ["USER_AGENT"] = "Mozilla/5.0"
        loader = WebBaseLoader(data.url)
        docs = loader.load()
        content = "\n".join([d.page_content for d in docs])
        # retrieve KB context
        kb_ctx_docs = retrieve(data.question, k=3)
        kb_context = "\n\n".join([d.page_content for d in kb_ctx_docs]) if kb_ctx_docs else ""
        prompt = f"Context:\n{content}\n\nKB:\n{kb_context}\n\nQuestion: {data.question}"
        try:
            payload = {"model": OLLAMA_TEXT_MODEL, "messages":[{"role":"user","content":prompt}]}
            resp = requests.post(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat", json=payload, timeout=60)
            if resp.ok:
                body = resp.json()
                answer = body.get("response") or body.get("text") or (body.get("choices")[0]["message"]["content"] if body.get("choices") else str(body))
            else:
                answer = f"LLM error: {resp.status_code}"
        except Exception as e:
            logger.exception("url_chat LLM error")
            answer = f"LLM call failed: {e}"
        return {"answer": answer}
    except Exception as e:
        logger.exception("chat_with_url error")
        return JSONResponse(status_code=500, content={"error": str(e)})
