import streamlit as st
import requests
import os
import json
import time
import base64
import uuid

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Multi-Modal RAG Q&A",
    layout="wide",
    page_icon="🤖"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_PREFIX = f"{BACKEND_URL}/api"

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>
/* Remove Streamlit’s default wide padding */
.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Headings */
h3, h4, h5, h6 {
    margin-bottom: 0.4rem !important;
}

/* Buttons */
div.stButton > button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    font-weight: 500;
    border: none;
}
div.stButton > button:hover {
    background-color: #1d4ed8 !important;
}

/* Subheader and Labels */
.stSubheader {
    margin-bottom: 0.3rem !important;
}
.stTextArea textarea {
    min-height: 100px !important;
    border-radius: 6px !important;
}

/* Separator Line */
hr, .stHorizontalBlock {
    margin: 0.5rem 0 !important;
}

/* Chat Bubbles */
.user-bubble {
    background: #0f172a;
    color: #f8fafc;
    padding: 10px 12px;
    border-radius: 10px;
    margin-bottom: 4px;
}
.assistant-bubble {
    background: #1e293b;
    color: #e2e8f0;
    padding: 10px 12px;
    border-radius: 10px;
    margin-bottom: 6px;
}

/* Sidebar cleanup */
.css-1d391kg, .css-18e3th9, header, footer {
    padding: 0 !important;
    margin: 0 !important;
}
header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session Initialization
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Header
# -----------------------------
st.title("🤖 Multi-Modal RAG Assistant")
st.caption("Ask questions from your documents, images, or audio (Powered by Local Ollama)")

col1, col2 = st.columns([2, 1])

# -----------------------------
# Left Section: Chat and KB Upload
# -----------------------------
with col1:
    st.subheader("💬 Chat")

    kb_files = st.file_uploader(
        "Upload a document to KB (optional)",
        type=["pdf", "docx", "txt", "csv"],
        key="kb_up"
    )

    if kb_files and st.button("📤 Upload to KB"):
        files = {"file": (kb_files.name, kb_files.getvalue())}
        with st.spinner("Uploading to Knowledge Base..."):
            resp = requests.post(f"{API_PREFIX}/add-to-kb", files=files)
        if resp.ok:
            st.success("✅ Uploaded to KB successfully!")
            try:
                st.json(resp.json())
            except:
                st.write(resp.text)
        else:
            st.error(resp.text)

    question = st.text_area("💭 Ask your question", height=120)
    ask = st.button("🚀 Ask")

    if ask:
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            if kb_files:
                files = {"file": (kb_files.name, kb_files.getvalue())}
                data = {"question": question, "session_id": st.session_state.session_id}
                try:
                    with st.spinner("Processing and generating answer..."):
                        resp = requests.post(f"{API_PREFIX}/file-chat", files=files, data=data, stream=True, timeout=300)
                    if resp.status_code == 200:
                        full = ""
                        answer_box = st.empty()
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            decoded = line.strip()
                            if decoded.startswith("data:"):
                                payload = decoded[len("data:"):].strip()
                                if payload == "[DONE]":
                                    break
                                try:
                                    d = json.loads(payload)
                                    token = d.get("content") or d.get("token") or ""
                                except:
                                    token = payload
                                full += token
                                answer_box.markdown(f"<div class='assistant-bubble'>{full}</div>", unsafe_allow_html=True)
                        st.session_state.history.append({"user": question, "assistant": full, "ts": time.time()})
                    else:
                        st.error(f"Backend error: {resp.status_code} {resp.text}")
                except Exception as e:
                    st.error(f"⚠️ Connection error: {e}")
            else:
                try:
                    with st.spinner("Getting response..."):
                        url = f"{API_PREFIX}/chat-stream"
                        with requests.post(url, json={"question": question}, stream=True, timeout=300) as r:
                            if r.status_code != 200:
                                st.error(f"❌ Streaming endpoint error {r.status_code}: {r.text}")
                            else:
                                full = ""
                                answer_box = st.empty()
                                for line in r.iter_lines(decode_unicode=True):
                                    if not line:
                                        continue
                                    decoded = line.strip()
                                    if decoded.startswith("data:"):
                                        payload = decoded[len("data:"):].strip()
                                        if payload == "[DONE]":
                                            break
                                        try:
                                            d = json.loads(payload)
                                            token = d.get("token") or d.get("content") or ""
                                        except:
                                            token = payload
                                        full += token
                                        answer_box.markdown(f"<div class='assistant-bubble'>{full}</div>", unsafe_allow_html=True)
                                st.session_state.history.append({"user": question, "assistant": full, "ts": time.time()})
                except Exception as e:
                    st.error(f"⚠️ Connection error: {e}")

# -----------------------------
# Right Section: Tools (OCR + Audio)
# -----------------------------
with col2:
    st.subheader("🧰 Tools")

    st.markdown("#### 🖼 OCR (Image → Text)")
    img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="ocr")
    if img and st.button("🔍 Extract Text"):
        files = {"file": (img.name, img.getvalue())}
        with st.spinner("Extracting text..."):
            resp = requests.post(f"{API_PREFIX}/extract-text-from-image", files=files)
        if resp.ok:
            data = resp.json()
            st.success("✅ Extracted text successfully.")
            st.text_area("📄 OCR Result", data.get("answer", ""), height=180)
        else:
            st.error(f"OCR Error: {resp.status_code} {resp.text}")

    st.markdown("#### 🎧 Audio → Text (Single)")
    aud = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"], key="audio")
    if aud and st.button("🎙 Transcribe Audio"):
        files = {"file": (aud.name, aud.getvalue())}
        with st.spinner("Transcribing audio..."):
            resp = requests.post(f"{API_PREFIX}/transcribe-audio", files=files)
        if resp.ok:
            data = resp.json()
            st.success("✅ Transcribed successfully.")
            st.text_area("🗒 Transcript", data.get("answer", ""), height=180)
        else:
            st.error(f"Transcribe Error: {resp.status_code} {resp.text}")

    st.markdown("#### 🎤 Audio → Text (Live)")
    st.caption("Use for short recordings. Requires microphone access.")
    try:
        from streamlit_mic_recorder import audio_recorder
        rec = audio_recorder()
        if rec and st.button("Send Recorded Clip"):
            b64 = base64.b64encode(rec).decode("utf-8")
            sess = st.session_state.session_id
            with st.spinner("Sending recorded audio..."):
                resp = requests.post(f"{API_PREFIX}/transcribe-stream", data={"session_id": sess, "chunk_b64": b64, "final": True})
            if resp.ok:
                data = resp.json()
                st.success("✅ Live transcript ready.")
                st.text_area("Live Transcript", data.get("answer", ""), height=160)
            else:
                st.error(resp.text)
    except Exception:
        st.info("⚠️ Install `streamlit-mic-recorder` for live recording support.")

# -----------------------------
# Conversation History
# -----------------------------
st.markdown("---")
st.subheader("🕒 Conversation History")

if st.session_state.history:
    for m in reversed(st.session_state.history):
        st.markdown(f"<div class='user-bubble'><b>You:</b> {m['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='assistant-bubble'><b>Assistant:</b> {m['assistant']}</div>", unsafe_allow_html=True)
else:
    st.info("No conversations yet.")
