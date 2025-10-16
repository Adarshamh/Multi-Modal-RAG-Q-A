import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from backend.core.text_extractor import extract_text_from_file
from backend.core.rag_engine import add_document_to_index
from backend.core.logger import logger
from backend.core.config import UPLOAD_DIR

router = APIRouter()

@router.post("/add-to-kb")
async def add_to_kb(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        text = extract_text_from_file(file_path)
        n = add_document_to_index(file.filename, text)
        return {"message": f"Document '{file.filename}' added to KB ({n} chunks)."}
    except Exception as e:
        logger.exception("add_to_kb error")
        return JSONResponse(status_code=500, content={"error": str(e)})
