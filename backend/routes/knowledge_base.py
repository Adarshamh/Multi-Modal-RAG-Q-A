import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..core.rag_engine import add_document_to_index
from ..core.text_extractor import extract_text_from_file
from ..core.logger import logger
from ..core.config import UPLOAD_DIR, MAX_FILE_SIZE_MB

router = APIRouter()

@router.post("/add-to-kb")
async def add_to_kb(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return JSONResponse(status_code=400, content={"error": "File too large."})
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(contents)
        text = extract_text_from_file(path)
        if not text:
            return JSONResponse(status_code=200, content={"ok": False, "message": "No text extracted."})
        added = add_document_to_index(file.filename, text)
        return {"ok": True, "added_chunks": added}
    except Exception as e:
        logger.exception("add-to-kb error")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
