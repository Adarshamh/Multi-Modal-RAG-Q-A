from fastapi import FastAPI, UploadFile, File, Form, WebSocket
from fastapi.responses import JSONResponse
import os, io, uuid
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import whisper

from memory import memory
from llm_utils import ask_llm
from knowledge_base import create_knowledge_base, retrieve_from_kb
from websocket import active_sessions, broadcast

from docx import Document
import pandas as pd
import PyPDF2

load_dotenv()
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 10))
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

app = FastAPI()

# ---------------- File text extraction ----------------
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".txt":
        return open(file_path, "r", encoding="utf-8").read()
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    elif ext == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    else:
        return "❌ Unsupported file type."

# ---------------- Chat with file ----------------
@app.post("/chat-with-file")
async def chat_with_file(file: UploadFile = File(...), question: str = Form(...), session_id: str = Form(default=str(uuid.uuid4()))):
    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_MB*1024*1024:
            return JSONResponse(status_code=400, content={"error": "File too large"})
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f: f.write(contents)
        file_content = extract_text_from_file(file_path)
        kb = create_knowledge_base(file_content)
        context = retrieve_from_kb(kb, question)
        memory.add_message(session_id, "user", question)
        answer = ask_llm(memory.get_session(session_id), question, context)
        memory.add_message(session_id, "assistant", answer)
        return {"answer": answer, "session_id": session_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------- Audio transcription ----------------
@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...), session_id: str = Form(default=str(uuid.uuid4()))):
    try:
        contents = await file.read()
        audio_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(audio_path, "wb") as f: f.write(contents)
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        memory.add_message(session_id, "user", f"Audio: {file.filename}")
        memory.add_message(session_id, "assistant", result["text"])
        return {"answer": result["text"], "session_id": session_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------- Image text extraction ----------------
@app.post("/extract-text-from-image")
async def extract_text_from_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(image)
        return {"answer": text.strip() or "⚠️ No text extracted."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------- WebSocket for real-time collaboration ----------------
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in active_sessions:
        active_sessions[session_id] = []
    active_sessions[session_id].append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            answer = ask_llm(memory.get_session(session_id), data)
            memory.add_message(session_id, "user", data)
            memory.add_message(session_id, "assistant", answer)
            for ws in active_sessions[session_id]:
                await ws.send_text(answer)
    except Exception:
        active_sessions[session_id].remove(websocket)
