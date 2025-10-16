import os, hashlib
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from backend.core.config import UPLOAD_DIR, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS, OLLAMA_MODEL
from backend.core.text_extractor import extract_text_from_file
from backend.core.rag_engine import add_document_to_index, retrieve
from backend.core.model_selector import select_model
from backend.core.logger import logger
import ollama

router = APIRouter()

def validate_file(fname: str):
    ext = os.path.splitext(fname)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@router.post("/file-chat")
async def chat_with_file(file: UploadFile = File(...), question: str = Form(...)):
    try:
        if not validate_file(file.filename):
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        # save file
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        text = extract_text_from_file(file_path)
        if not text:
            return {"answer": "⚠️ Could not extract text from file."}
        added = add_document_to_index(file.filename, text)
        docs = retrieve(question, k=3)
        context = "\n\n".join([d["page_content"] for d in docs]) if docs else text[:3000]
        prompt = f"Use the context below to answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
        model = select_model("text")
        try:
            resp = ollama.chat(model=model, messages=[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":prompt}])
            if isinstance(resp, dict):
                answer = resp.get("text") or (resp.get("choices")[0]["message"]["content"] if resp.get("choices") else None)
            else:
                answer = str(resp)
        except Exception as e:
            logger.exception("Ollama error")
            return {"answer": f"LLM error: {e}"}
        return {"answer": answer, "rag_chunks_indexed": added}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("file_chat error")
        return JSONResponse(status_code=500, content={"error": str(e)})
