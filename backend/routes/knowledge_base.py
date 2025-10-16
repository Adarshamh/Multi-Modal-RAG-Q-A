import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..core.text_extractor import extract_text_from_file
from ..core.rag_engine import add_document_to_index
from ..core.config import UPLOAD_DIR
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
            return JSONResponse(status_code=400, content={"error": "No text extracted from file."})
        n = add_document_to_index(file.filename, text)
        return {"message": f"Document '{file.filename}' added to KB ({n} chunks)."}
    except Exception as e:
        logger.exception("add_to_kb error")
        return JSONResponse(status_code=500, content={"error": str(e)})
