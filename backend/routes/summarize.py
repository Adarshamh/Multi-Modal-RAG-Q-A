from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..core.text_extractor import extract_text_from_file
from ..core.config import OLLAMA_HOST, OLLAMA_PORT, OLLAMA_TEXT_MODEL
from ..core.logger import logger
import requests, os

router = APIRouter()

@router.post("/auto-summarize")
async def summarize_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        tmp_path = f"/tmp/{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(contents)
        text = extract_text_from_file(tmp_path)
        if not text:
            return {"answer":"No content to summarize."}
        prompt = f"Summarize the following text in bullet points:\n\n{text[:4000]}"
        try:
            payload = {"model": OLLAMA_TEXT_MODEL, "messages": [{"role":"user","content":prompt}]}
            resp = requests.post(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat", json=payload, timeout=60)
            if resp.ok:
                body = resp.json()
                ans = body.get("response") or body.get("text") or (body.get("choices")[0]["message"]["content"] if body.get("choices") else str(body))
            else:
                ans = f"LLM error: {resp.status_code}"
        except Exception as e:
            logger.exception("summarize LLM error")
            ans = f"LLM call failed: {e}"
        return {"answer": ans}
    except Exception as e:
        logger.exception("summarize error")
        return JSONResponse(status_code=500, content={"error": str(e)})
