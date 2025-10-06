from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from core.text_extractor import extract_text
from core.embedding_manager import add_to_kb
from core.config import UPLOAD_DIR
import os

router = APIRouter()

@router.post("/add-to-kb")
async def add_kb(file: UploadFile):
    try:
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        text = extract_text(file_path)
        add_to_kb(text)
        return {"message": f"Document '{file.filename}' added to knowledge base."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
