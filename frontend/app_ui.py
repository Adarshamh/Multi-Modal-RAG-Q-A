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
# Typography & Styling
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    background-color: #f9fafb;
}

/* Title and Caption */
h1 {
    color: #1e3a8a !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    text-align: center;
    letter-spacing: -0.5px;
}
.caption, .stCaption {
    text-align: center !important;
    font-size: 1.05rem !important;
    color: #475569 !important;
    margin-bottom: 1.5rem !important;
}

/* Tabs */
div[data-testid="stTabs"] button {
    background-color: #eff6ff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: #1e40af !important;
    font-size: 1rem !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #2563eb !important;
    color: white !important;
    font-weight: 700 !important;
}

/* Section Titles */
h2, h3, h4 {
    color: #1e293b !important;
    font-weight: 700 !important;
}
h2 { font-size: 1.6rem !important; margin-bottom: 0.6rem; }
h3 { font-size: 1.3rem !important; margin-bottom: 0.4rem; }

/* Labels and Inputs */
label, .stTextInput label, .stTextArea label {
    font-size: 1rem !important;
    font-weight: 500 !important;
    color: #1e293b !important;
}

/* Text Areas */
.stTextArea textarea {
    min-height: 110px !important;
    border-radius: 8px !important;
    border: 1px solid #dbeafe !important;
    background: #f8fafc !important;
    font-size: 1rem !important;
    padding: 0.6rem !important;
    color: #111827 !important;
}

/* Buttons */
div.stButton > button {
    background-color: #2563eb !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.2rem !important;
    font-size: 1rem !important;
    border: none;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background-color: #1d4ed8 !important;
    transform: scale(1.04);
}

/* Chat Bubbles */
.user-bubble {
    background: #1e3a8a;
    color: #f8fafc;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 8px;
    font-size: 1rem;
    line-height: 1.45;
}
.assistant-bubble {
    background: #e2e8f0;
    color: #1e293b;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-size: 1rem;
    line-height: 1.45;
}

/* Alerts */
.stAlert > div {
    font-size: 1rem !important;
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
st.title("🤖 Multi-Modal RAG Q&A System")
st.caption("Ask questions from your documents, images, or audio (Powered by Local Ollama)")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["💬 Chat Q&A", "🖼️ Image → Text (OCR)", "🎧 Audio → Text"])

# -----------------------------
# TAB 1: Chat Q&A (Original Logic Unchanged)
# -----------------------------
with tab1:
    st.subheader("💬 Chat with your Knowledge Base")

    kb_files = st.file_uploader(
        "Upload a document to KB (optional)",
        type=["pdf", "docx", "txt", "csv"],
        key="kb_up"
    )

    if kb_files and st.button("📤 Upload to KB", key="upload_kb"):
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

    st.markdown("---")
    st.subheader("🕒 Conversation History")

    if st.session_state.history:
        for m in reversed(st.session_state.history):
            st.markdown(f"<div class='user-bubble'><b>You:</b> {m['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='assistant-bubble'><b>Assistant:</b> {m['assistant']}</div>", unsafe_allow_html=True)
    else:
        st.info("No conversations yet.")

# -----------------------------
# TAB 2: Image → Text (OCR)
# -----------------------------
with tab2:
    st.subheader("🖼️ Extract Text from Image")

    img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="ocr")
    if img and st.button("🔍 Extract Text", key="ocr_btn"):
        files = {"file": (img.name, img.getvalue())}
        with st.spinner("Extracting text..."):
            resp = requests.post(f"{API_PREFIX}/extract-text-from-image", files=files)
        if resp.ok:
            data = resp.json()
            st.success("✅ Text extracted successfully.")
            st.text_area("📄 OCR Result", data.get("answer", ""), height=180)
        else:
            st.error(f"OCR Error: {resp.status_code} {resp.text}")

# -----------------------------
# TAB 3: Audio → Text
# -----------------------------
with tab3:
    st.subheader("🎧 Audio Transcription")

    aud = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"], key="audio")
    if aud and st.button("🎙 Transcribe Audio", key="audio_btn"):
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
        if rec and st.button("Send Recorded Clip", key="live_audio_btn"):
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
