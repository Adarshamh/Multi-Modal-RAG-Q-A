import os
import hashlib
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ..core.config import UPLOAD_DIR, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS
from ..core.text_extractor import extract_text_from_file
from ..core.rag_engine import add_document_to_index, retrieve
from ..core.config import OLLAMA_HOST, OLLAMA_PORT, OLLAMA_TEXT_MODEL
from ..core.logger import logger
import requests, json

router = APIRouter()

def validate_file(fname: str):
    ext = os.path.splitext(fname)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@router.post("/file-chat")
async def chat_with_file(file: UploadFile = File(...), question: str = Form(...), template: str = Form("qa"), session_id: str = Form(None)):
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
            return {"answer": "⚠️ Could not extract text from file."}
        # add to RAG index
        added = add_document_to_index(file.filename, text)
        # retrieve context
        docs = retrieve(question, k=3)
        context = "\n\n".join([d.page_content for d in docs]) if docs else text[:3000]
        prompt = f"Use the context below to answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
        # Ollama HTTP call
        try:
            payload = {"model": OLLAMA_TEXT_MODEL, "messages": [{"role":"user","content": prompt}]}
            r = requests.post(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat", json=payload, timeout=60)
            if r.ok:
                body = r.json()
                answer = body.get("response") or body.get("text") or (body.get("choices")[0]["message"]["content"] if body.get("choices") else str(body))
            else:
                answer = f"LLM HTTP error: {r.status_code}"
        except Exception as e:
            logger.exception("Ollama chat error")
            answer = f"LLM error: {e}"
        return {"answer": answer, "rag_chunks_indexed": added}
    except Exception as e:
        logger.exception("file_chat error")
        return JSONResponse(status_code=500, content={"error": str(e)})
