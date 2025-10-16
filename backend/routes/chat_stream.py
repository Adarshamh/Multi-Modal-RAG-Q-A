from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.core.logger import logger
from backend.core.model_selector import select_model
from backend.core.config import OLLAMA_HOST, OLLAMA_PORT
import requests, json, time, os

router = APIRouter()

class StreamQuery(BaseModel):
    question: str

def _ollama_stream_payload(model, prompt):
    """
    Ollama streaming: some Ollama builds support streaming endpoints.
    We'll attempt to call /api/chat/stream; if not available we'll fallback to /api/chat.
    """
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}

@router.post("/chat-stream")
async def chat_stream(data: StreamQuery):
    try:
        def event_generator():
            try:
                prompt = data.question
                model = select_model("text")
                # attempt Ollama streaming endpoint first
                stream_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat/stream"
                plain_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat"
                # Try streaming with requests (server may choose to send newline-delimited JSON)
                try:
                    payload = _ollama_stream_payload(model, prompt)
                    with requests.post(stream_url, json=payload, stream=True, timeout=120) as r:
                        if r.status_code == 200:
                            # Stream lines from Ollama directly if available
                            for raw in r.iter_lines(decode_unicode=True):
                                if raw is None:
                                    continue
                                line = raw.strip()
                                if not line:
                                    continue
                                # Ollama may send JSON per line or token strings
                                try:
                                    data_json = json.loads(line)
                                    # try to extract token/text
                                    token = data_json.get("delta") or data_json.get("token") or data_json.get("text") or data_json.get("response") or data_json.get("content") or ""
                                    if isinstance(token, dict):
                                        token = token.get("content", "")
                                except Exception:
                                    token = line
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            return
                        else:
                            logger.info(f"Ollama stream responded {r.status_code}; falling back")
                except Exception as e:
                    logger.info(f"Ollama streaming unavailable: {e}; falling back to non-streaming call")

                # fallback: call non-streaming endpoint and emulate streaming
                try:
                    payload_plain = {"model": model, "messages": [{"role": "user", "content": prompt}]}
                    r2 = requests.post(plain_url, json=payload_plain, timeout=120)
                    if r2.ok:
                        try:
                            body = r2.json()
                        except Exception:
                            body = {"text": r2.text}
                        text = body.get("response") or body.get("text") or (body.get("choices")[0]["message"]["content"] if body.get("choices") else str(body))
                    else:
                        text = f"LLM error: {r2.status_code}"
                except Exception as e:
                    logger.exception("chat_stream LLM call error")
                    text = f"LLM error: {e}"

                # simple token-emulation streaming
                for i in range(0, len(text), 8):
                    token = text[i:i+8]
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    time.sleep(0.03)
            except Exception as e:
                logger.exception("chat_stream generation error")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        logger.exception("chat_stream error")
        raise HTTPException(status_code=500, detail=str(e))
