import os
import json
import requests
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from ..core.config import (
    UPLOAD_DIR, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS,
    OLLAMA_HOST, OLLAMA_PORT, OLLAMA_TEXT_MODEL
)
from ..core.text_extractor import extract_text_from_file
from ..core.rag_engine import add_document_to_index, retrieve
from ..core.logger import logger

router = APIRouter()


def validate_file(fname: str):
    ext = os.path.splitext(fname)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@router.post("/file-chat")
async def chat_with_file(
    file: UploadFile = File(...),
    question: str = Form(...),
    template: str = Form("qa"),
    session_id: str = Form(None)
):
    try:
        # ---- 1. Validate file ----
        if not validate_file(file.filename):
            return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return JSONResponse(status_code=400, content={"error": "File too large."})

        # ---- 2. Save uploaded file ----
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        # ---- 3. Extract text ----
        text = extract_text_from_file(file_path)
        if not text:
            return JSONResponse(status_code=400, content={"error": "⚠️ Could not extract text from file."})

        # ---- 4. Add to RAG index ----
        add_document_to_index(file.filename, text)

        # ---- 5. Retrieve context ----
        docs = retrieve(question, k=3)
        context = "\n\n".join([d.page_content for d in docs]) if docs else text[:3000]

        prompt = (
            f"Use the context below to answer accurately.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}"
        )

        payload = {
            "model": OLLAMA_TEXT_MODEL,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }

        ollama_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat"

        # ---- 6. Stream safely ----
        def ollama_stream():
            try:
                with requests.post(ollama_url, json=payload, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    buffer = ""

                    for raw_line in r.iter_lines():
                        if not raw_line:
                            continue

                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue

                        # Remove "data: " prefix if present
                        if line.startswith("data: "):
                            line = line[6:].strip()

                        # Stop signal
                        if line.lower() in ("[done]", "[done]."):
                            yield "data: [DONE]\n\n"
                            break

                        # ---- Handle possible malformed fragments ----
                        buffer += line
                        try:
                            event = json.loads(buffer)
                            content = event.get("message", {}).get("content", "")
                            if content:
                                yield f"data: {content}\n\n"
                            buffer = ""  # reset buffer after success
                        except json.JSONDecodeError:
                            # incomplete JSON fragment, keep buffering
                            continue

            except Exception as e:
                logger.exception("Streaming error from Ollama")
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(ollama_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.exception("file_chat error")
        return JSONResponse(status_code=500, content={"error": str(e)})
