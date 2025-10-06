import os
import io
import hashlib
import logging
import asyncio
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import whisper
from docx import Document
import pandas as pd
import PyPDF2
from moviepy.editor import AudioFileClip

from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import WebBaseLoader

# Attempt Ollama first, fallback to OpenAI GPT
try:
    from ollama import Ollama
    ollama_available = True
    ollama_client = Ollama()
except:
    ollama_available = False
    import openai

load_dotenv()

# -------------------- CONFIG --------------------
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".csv", ".png", ".jpg", ".jpeg", ".mp3", ".wav", ".mp4", ".mov", ".py", ".js", ".cs"}

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
ffmpeg_path = os.getenv("FFMPEG_PATH")
if ffmpeg_path:
    os.environ["PATH"] += os.pathsep + ffmpeg_path

# Logging
logging.basicConfig(filename="app.log", level=logging.INFO)

# Initialize FastAPI
app = FastAPI()

# -------------------- MEMORY / CACHE --------------------
session_memory = {}  # {session_id: [{"role":"user","content":...}, ...]}
cache = {}  # {cache_key: answer}

def get_session_messages(session_id):
    return session_memory.get(session_id, [])

def append_session_message(session_id, role, content):
    session_memory.setdefault(session_id, []).append({"role": role, "content": content})

def compute_cache_key(file_path, question):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return f"{file_hash}:{question}"

# -------------------- PROMPT TEMPLATES --------------------
prompt_templates = {
    "qa": "Answer the user's question based on the content:\n{content}\nQuestion: {question}",
    "summary": "Summarize the following content in bullet points:\n{content}",
    "code_review": "Review the following code and provide suggestions:\n{content}"
}

def apply_template(template_name, content, question=""):
    template = prompt_templates.get(template_name, "{content}\nQuestion: {question}")
    return template.format(content=content, question=question)

# -------------------- FILE HANDLING --------------------
def validate_file(file_name):
    ext = os.path.splitext(file_name)[-1].lower()
    return ext in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    elif ext == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext in [".py", ".js", ".cs"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return None

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -------------------- HYBRID LLM --------------------
def chat_with_llm(messages):
    try:
        # Ollama offline
        if ollama_available:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            response = ollama_client.completion(prompt=prompt, model="llama3-8b")
            return response.text
        else:  # fallback OpenAI GPT
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return response.choices[0].message.content
    except Exception as e:
        logging.error(f"LLM Error: {str(e)}")
        return "⚠️ LLM Error. See logs."

# -------------------- AUDIO / IMAGE --------------------
whisper_model = whisper.load_model("base")

def transcribe_audio_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".mp4", ".mov"]:
        clip = AudioFileClip(file_path)
        audio_path = file_path.replace(ext, ".wav")
        clip.audio.write_audiofile(audio_path)
        file_path = audio_path
    result = whisper_model.transcribe(file_path)
    return result.get("text", "")

def extract_text_from_image_file(file_path):
    image = Image.open(file_path)
    return pytesseract.image_to_string(image).strip()

# -------------------- RAG VECTOR STORE --------------------
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = None
kb_docs = []

def add_to_kb(file_content):
    global vector_store
    from langchain.docstore.document import Document as LDoc
    doc = LDoc(page_content=file_content)
    kb_docs.append(doc)
    vector_store = FAISS.from_documents(kb_docs, embeddings)

def retrieve_from_kb(question, top_k=5):
    if not vector_store:
        return ""
    docs = vector_store.similarity_search(question, k=top_k)
    return "\n".join([d.page_content for d in docs])

# -------------------- FASTAPI ENDPOINTS --------------------
class URLQuery(BaseModel):
    url: str
    question: str
    template: str = "qa"
    session_id: str = None

@app.post("/chat-with-file")
async def chat_file(file: UploadFile = File(...), question: str = Form(...), template: str = Form("qa"), session_id: str = Form(None)):
    try:
        if not validate_file(file.filename):
            return JSONResponse(status_code=400, content={"error": "❌ Unsupported file type."})
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return JSONResponse(status_code=400, content={"error": "❌ File too large."})
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        # caching
        key = compute_cache_key(file_path, question)
        if key in cache:
            return {"answer": cache[key]}

        file_content = extract_text_from_file(file_path)
        if not file_content:
            return {"answer": "⚠️ Could not extract content."}

        # chunking
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
        logging.error(str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat-with-url")
async def chat_url(data: URLQuery):
    try:
        os.environ["USER_AGENT"] = "Mozilla/5.0"
        loader = WebBaseLoader(data.url)
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])
        kb_context = retrieve_from_kb(data.question)
        prompt = apply_template(data.template, content + "\n\n" + kb_context, data.question)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if data.session_id:
            messages += get_session_messages(data.session_id)
        messages.append({"role": "user", "content": prompt})
        ans = chat_with_llm(messages)
        if data.session_id:
            append_session_message(data.session_id, "user", prompt)
            append_session_message(data.session_id, "assistant", ans)
        return {"answer": ans}
    except Exception as e:
        logging.error(str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/extract-text-from-image")
async def extract_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        text = extract_text_from_image_file(file_path)
        return {"answer": text or "⚠️ No text extracted."}
    except Exception as e:
        logging.error(str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        text = transcribe_audio_file(file_path)
        return {"answer": text or "⚠️ No transcript returned."}
    except Exception as e:
        logging.error(str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/add-to-kb")
async def add_kb(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        text = extract_text_from_file(file_path)
        add_to_kb(text)
        return {"message": f"Document '{file.filename}' added to knowledge base."}
    except Exception as e:
        logging.error(str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------- RUN SERVER --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
