from fastapi import APIRouter, Form
from ..core.logger import logger
from ..core.rag_engine import retrieve
from ..core.model_selector import select_model
from ..core.config import OLLAMA_HOST, OLLAMA_PORT
import requests

router = APIRouter()

@router.post("/auto-summarize")
def summarize_text(query: str = Form(...), topk: int = 5):
    """
    Simple summarization using retrieved context + Ollama.
    """
    try:
        docs = retrieve(query, k=topk)
        context = "\n\n".join([d.get("text", "") for d in docs])
        prompt = f"Summarize the following context:\n\n{context}\n\nSummary:"
        model = select_model("text")
        r = requests.post(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat", json={"model": model, "messages":[{"role":"user","content":prompt}]}, timeout=60)
        if r.ok:
            try:
                return r.json()
            except:
                return {"answer": r.text}
        else:
            return {"error": r.text}
    except Exception as e:
        logger.exception("summarize error")
        return {"error": str(e)}
