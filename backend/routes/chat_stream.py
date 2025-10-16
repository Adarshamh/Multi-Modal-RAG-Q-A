from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.core.logger import logger
from backend.core.model_selector import select_model
import requests, json, os

router = APIRouter()

class StreamQuery(BaseModel):
    question: str

@router.post("/chat-stream")
async def chat_stream(data: StreamQuery):
    """
    Handles live streaming chat responses from Ollama or compatible LLM.
    """
    try:
        def event_generator():
            try:
                prompt = data.question
                model = select_model("text")

                # ✅ Enable true streaming
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                }

                # Call Ollama API with streaming enabled
                response = requests.post(
                    f"http://{os.getenv('OLLAMA_HOST','127.0.0.1')}:{os.getenv('OLLAMA_PORT','11434')}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=800
                )

                if not response.ok:
                    yield f"data: {json.dumps({'error': f'LLM error: {response.status_code}'})}\n\n"
                    return

                # ✅ Properly process each JSON line (Ollama streams line-delimited JSON)
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data_chunk = json.loads(line.decode("utf-8"))
                        token = data_chunk.get("message", {}).get("content", "")
                        if token:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed chunk: {line}")
                        continue

                # Stream end
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("chat_stream generation error")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # Return a streaming response to frontend
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.exception("chat_stream error")
        raise HTTPException(status_code=500, detail=str(e))
