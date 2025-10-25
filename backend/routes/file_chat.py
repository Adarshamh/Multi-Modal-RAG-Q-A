import os
import json
import requests
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from ..core.config import UPLOAD_DIR, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS, OLLAMA_HOST, OLLAMA_PORT, OLLAMA_TEXT_MODEL
from ..core.text_extractor import extract_text_from_file
from ..core.rag_engine import add_document_to_index, retrieve
from ..core.logger import logger
from ..core.model_selector import select_model

router = APIRouter()

def validate_file(fname: str):
    ext = os.path.splitext(fname)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@router.post("/file-chat")
async def chat_with_file(file: UploadFile = File(...), question: str = Form(...), template: str = Form("qa"), session_id: str = Form(None)):
    """
    Upload a file, add to index, query RAG and stream a response via Ollama.
    Streams SSE events as JSON objects so the frontend can accumulate them.
    """
    try:
        if not validate_file(file.filename):
            return JSONResponse(status_code=400, content={"error": "Unsupported file type."})
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return JSONResponse(status_code=400, content={"error": "File too large."})
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        text = extract_text_from_file(file_path)
        if not text:
            return JSONResponse(status_code=200, content={"answer": "⚠️ Could not extract text from file."})

        # Add to RAG index
        add_document_to_index(file.filename, text)

        # Retrieve best chunks for context
        docs = retrieve(question, k=3)
        context = "\n\n".join([d.get("text", d.get("page_content", "")) for d in docs]) if docs else text[:3000]
        prompt = f"Use the context below to answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"

        model = select_model("text") or OLLAMA_TEXT_MODEL
        payload = {"model": model, "stream": True, "messages": [{"role":"user","content": prompt}]}
        ollama_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat"

        def ollama_stream():
            try:
                with requests.post(ollama_url, json=payload, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    for raw in r.iter_lines():
                        if not raw:
                            continue
                        # Handle bytes or string safely
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
                            ev = json.loads(payload_text)
                            token = ev.get("message", {}).get("content") or ev.get("response") or ev.get("text") or payload_text
                        except Exception:
                            token = payload_text
                        yield f"data: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'content': ''})}\n\n"
            except Exception as e:
                logger.exception("Streaming error from Ollama")
                yield f"data: {json.dumps({'content': f'[ERROR] {str(e)}'})}\n\n"
        return StreamingResponse(ollama_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.exception("file_chat error")
        return JSONResponse(status_code=500, content={"error": str(e)})
