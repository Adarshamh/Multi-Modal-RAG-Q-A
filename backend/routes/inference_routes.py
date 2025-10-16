from fastapi import APIRouter
from pydantic import BaseModel
from ..core.model_selector import select_model
from ..core.config import OLLAMA_HOST, OLLAMA_PORT
from ..core.logger import logger
import requests, json

router = APIRouter()

class InferRequest(BaseModel):
    input_type: str  # text/file/url
    prompt: str

@router.post("/infer")
def infer(req: InferRequest):
    try:
        model = select_model("text")
        payload = {"model": model, "messages": [{"role":"user","content": req.prompt}]}
        r = requests.post(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat", json=payload, timeout=60)
        if r.ok:
            try:
                body = r.json()
            except Exception:
                body = {"text": r.text}
            answer = body.get("response") or body.get("text") or (body.get("choices")[0]["message"]["content"] if body.get("choices") else str(body))
            return {"answer": answer}
        else:
            return {"error": f"LLM error: {r.status_code}"}
    except Exception as e:
        logger.exception("infer error")
        return {"error": str(e)}
