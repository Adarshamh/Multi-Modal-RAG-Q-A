import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..core.config import UPLOAD_DIR
from ..core.text_extractor import extract_text_from_file
from ..core.rag_engine import add_document_to_index
from ..core.logger import logger

router = APIRouter()

@router.post("/add-to-kb")
async def add_to_kb(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        text = extract_text_from_file(file_path)
        if not text:
            return JSONResponse(status_code=400, content={"ok": False, "error": "Could not extract text from file."})
        added = add_document_to_index(file.filename, text)
        return {"ok": True, "added_chunks": added, "filename": file.filename}
    except Exception as e:
        logger.exception("add-to-kb error")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
