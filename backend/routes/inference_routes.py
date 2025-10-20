from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from ..core.logger import logger
from ..core.config import OLLAMA_HOST, OLLAMA_PORT, OLLAMA_TEXT_MODEL
import requests

router = APIRouter()

@router.post("/infer")
def infer(input_type: str = Form(...), prompt: str = Form(...)):
    """
    Simple non-streamed inference endpoint.
    """
    try:
        payload = {"model": OLLAMA_TEXT_MODEL, "messages": [{"role":"user","content": prompt}]}
        r = requests.post(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat", json=payload, timeout=120)
        if r.ok:
            try:
                body = r.json()
                answer = body.get("response") or body.get("text") or (body.get("choices")[0]["message"]["content"] if body.get("choices") else r.text)
            except Exception:
                answer = r.text
            return {"ok": True, "answer": answer}
        else:
            return JSONResponse(status_code=500, content={"ok": False, "error": f"LLM error {r.status_code}"})
    except Exception as e:
        logger.exception("infer error")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
