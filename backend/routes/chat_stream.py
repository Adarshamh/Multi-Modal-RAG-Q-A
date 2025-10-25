import json
import requests
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..core.logger import logger
from ..core.config import OLLAMA_HOST, OLLAMA_PORT, OLLAMA_TEXT_MODEL
from ..core.model_selector import select_model

router = APIRouter()

class StreamQuery(BaseModel):
    question: str

@router.post("/chat-stream")
async def chat_stream(payload: StreamQuery):
    """
    SSE-style streaming endpoint to proxy Ollama output.
    """
    try:
        prompt = payload.question
        model = select_model("text") or OLLAMA_TEXT_MODEL
        ollama_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat"
        req_payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}

        with requests.post(ollama_url, json=req_payload, stream=True, timeout=300) as r:
            r.raise_for_status()

            def event_gen():
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    # Convert bytes to string safely
                    line = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw.strip()
                    if not line:
                        continue

                    if line.startswith("data:"):
                        payload_text = line[len("data:"):].strip()
                    else:
                        payload_text = line

                    if payload_text == "[DONE]":
                        break

                    try:
                        obj = json.loads(payload_text)
                        token = obj.get("message", {}).get("content") or obj.get("response") or obj.get("text") or payload_text
                    except Exception:
                        token = payload_text

                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                    time.sleep(0.01)

                yield f"data: {json.dumps({'token': ''})}\n\n"

            return StreamingResponse(event_gen(), media_type="text/event-stream")

    except requests.exceptions.RequestException as e:
        logger.exception("Ollama connection failed")
        raise HTTPException(status_code=502, detail=f"Ollama connection failed: {str(e)}")

    except Exception as e:
        logger.exception("chat_stream error")
        raise HTTPException(status_code=500, detail=str(e))
