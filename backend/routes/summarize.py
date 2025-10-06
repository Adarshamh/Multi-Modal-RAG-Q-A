from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from core.text_extractor import extract_text
from core.rag_engine import chat_with_llm
import os

router = APIRouter()
UPLOAD_DIR = os.path.join(os.getcwd(), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

prompt_template = "Summarize the following content in bullet points:\n{content}"

@router.post("/summarize-file")
async def summarize_file(file: UploadFile, session_id: str = Form(None)):
    try:
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        text = extract_text(file_path)
        prompt = prompt_template.format(content=text)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        ans = chat_with_llm(messages + [{"role": "user", "content": prompt}])
        return {"answer": ans}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
