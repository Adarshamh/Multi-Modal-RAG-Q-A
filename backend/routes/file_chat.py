from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import os, hashlib
from core.text_extractor import extract_text
from core.rag_engine import chat_with_llm

router = APIRouter()

UPLOAD_DIR = os.path.join(os.getcwd(), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

session_memory = {}
cache = {}

def get_session_messages(session_id):
    return session_memory.get(session_id, [])

def append_session_message(session_id, role, content):
    session_memory.setdefault(session_id, []).append({"role": role, "content": content})

def compute_cache_key(file_path, question):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return f"{file_hash}:{question}"

prompt_templates = {
    "qa": "Answer the user's question based on the content:\n{content}\nQuestion: {question}",
    "summary": "Summarize the following content in bullet points:\n{content}",
    "code_review": "Review the following code and provide suggestions:\n{content}"
}

def apply_template(template_name, content, question=""):
    template = prompt_templates.get(template_name, "{content}\nQuestion: {question}")
    return template.format(content=content, question=question)

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@router.post("/chat-with-file")
async def chat_file(file: UploadFile, question: str = Form(...), template: str = Form("qa"), session_id: str = Form(None)):
    try:
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        key = compute_cache_key(file_path, question)
        if key in cache:
            return {"answer": cache[key]}
        file_content = extract_text(file_path)
        if not file_content:
            return {"answer": "⚠️ Could not extract content."}
        chunks = chunk_text(file_content)
        answers = []
        for chunk in chunks:
            prompt = apply_template(template, chunk, question)
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            if session_id:
                messages += get_session_messages(session_id)
            messages.append({"role": "user", "content": prompt})
            ans = chat_with_llm(messages)
            answers.append(ans)
            if session_id:
                append_session_message(session_id, "user", prompt)
                append_session_message(session_id, "assistant", ans)
        final_answer = "\n\n".join(answers)
        cache[key] = final_answer
        return {"answer": final_answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
