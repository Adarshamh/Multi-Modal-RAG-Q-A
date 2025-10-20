from fastapi import APIRouter, Form
from ..core.logger import logger
from ..core.rag_engine import retrieve
from ..core.config import OLLAMA_HOST, OLLAMA_PORT, OLLAMA_TEXT_MODEL
import requests

router = APIRouter()

@router.post("/auto-summarize")
def auto_summarize(question: str = Form(...)):
    """
    Very lightweight summarizer: fetch top chunks then ask LLM to summarize.
    """
    try:
        docs = retrieve(question or "summarize", k=5)
        context = "\n\n".join([d.get("text","") for d in docs]) or ""
        payload = {"model": OLLAMA_TEXT_MODEL, "messages": [{"role":"user", "content": f"Summarize the following:\n\n{context}"}]}
        r = requests.post(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat", json=payload, timeout=120)
        if r.ok:
            try:
                body = r.json()
                ans = body.get("response") or body.get("text") or r.text
            except Exception:
                ans = r.text
            return {"ok": True, "summary": ans}
        else:
            return {"ok": False, "error": f"LLM error {r.status_code}"}
    except Exception as e:
        logger.exception("auto-summarize error")
        return {"ok": False, "error": str(e)}
