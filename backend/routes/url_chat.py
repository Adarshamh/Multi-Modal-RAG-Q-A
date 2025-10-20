from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import requests
from bs4 import BeautifulSoup
from ..core.rag_engine import retrieve
from ..core.logger import logger

router = APIRouter()

@router.post("/chat-with-url")
def chat_with_url(payload: dict):
    url = payload.get("url")
    question = payload.get("question", "")
    if not url:
        raise HTTPException(status_code=400, detail="Missing url")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        text = soup.get_text(separator="\n")
        # optionally add to KB? Here we only use page text for immediate RAG
        docs = retrieve(question, k=3)
        context = "\n\n".join([d.get("text", "") for d in docs]) or text[:3000]
        # call Ollama sync for simplicity
        from ..core.config import OLLAMA_HOST, OLLAMA_PORT, OLLAMA_TEXT_MODEL
        payload = {"model": OLLAMA_TEXT_MODEL, "messages": [{"role":"user","content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}]}
        ollama = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat"
        resp = requests.post(ollama, json=payload, timeout=120)
        if resp.ok:
            try:
                body = resp.json()
                answer = body.get("response") or body.get("text") or (body.get("choices")[0]["message"]["content"] if body.get("choices") else resp.text)
            except Exception:
                answer = resp.text
            return {"ok": True, "answer": answer}
        else:
            return JSONResponse(status_code=500, content={"ok": False, "error": f"LLM error {resp.status_code}"})
    except Exception as e:
        logger.exception("chat-with-url error")
        raise HTTPException(status_code=500, detail=str(e))
