from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from backend.core.text_extractor import extract_text_from_file
from backend.core.model_selector import select_model
from backend.core.logger import logger
import ollama
import os

router = APIRouter()

@router.post("/auto-summarize")
async def summarize_file(file: UploadFile = File(...)):
    try:
        os.makedirs(os.path.join("data","uploads"), exist_ok=True)
        tmp_path = os.path.join("data","uploads", file.filename)
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        text = extract_text_from_file(tmp_path)
        if not text:
            return {"answer":"No content to summarize."}
        prompt = f"Summarize the following text in bullet points:\n\n{text[:4000]}"
        model = select_model("text")
        resp = ollama.chat(model=model, messages=[{"role":"user","content":prompt}])
        ans = resp.get("text") if isinstance(resp, dict) and resp.get("text") else str(resp)
        return {"answer": ans}
    except Exception as e:
        logger.exception("summarize_file error")
        return JSONResponse(status_code=500, content={"error": str(e)})
