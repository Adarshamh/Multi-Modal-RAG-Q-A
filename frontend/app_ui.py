import streamlit as st
import requests
import os
import json
import time
from io import BytesIO

st.set_page_config(page_title="Multi-Modal RAG Q&A", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_PREFIX = f"{BACKEND_URL}/api"

st.title("🤖 Multi-Modal RAG Assistant")
st.caption("Ask questions from your documents, images, audio or URLs (Local Ollama)")

# session
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("💬 Ask Question / File Chat")
    uploaded = st.file_uploader("Upload a document (optional) — if provided question is about this file", type=["pdf","docx","txt","png","jpg","jpeg","mp4","wav","csv"])
    question = st.text_area("Enter your question", height=140, placeholder="e.g., Summarize the uploaded document or answer questions about it.")
    ask = st.button("Ask")

    if ask:
        if uploaded:
            # send to file-chat endpoint with file
            files = {"file": (uploaded.name, uploaded.getvalue())}
            data = {"question": question or "Give a brief summary", "template": "qa", "session_id": st.session_state.session_id}
            try:
                with st.spinner("Uploading file & asking..."):
                    resp = requests.post(f"{API_PREFIX}/file-chat", files=files, data=data, timeout=2000)
                if resp.ok:
                    try:
                        out = resp.json()
                        st.success("Answer")
                        st.write(out.get("answer", out))
                        st.session_state.history.append({"user": question, "assistant": out.get("answer", ""), "ts": time.time()})
                    except Exception:
                        st.error("Invalid JSON response")
                else:
                    st.error(f"Backend error: {resp.status_code} {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")
        else:
            # stream via chat-stream
            try:
                with st.spinner("Contacting model..."):
                    # connect to SSE endpoint
                    url = f"{API_PREFIX}/chat-stream"
                    # use requests streaming
                    with requests.post(url, json={"question": question}, stream=True, timeout=2000) as r:
                        if r.status_code != 200:
                            st.error(f"Streaming endpoint returned {r.status_code}: {r.text}")
                        else:
                            full = ""
                            answer_box = st.empty()
                            for line in r.iter_lines(decode_unicode=True):
                                if not line:
                                    continue
                                decoded = line.strip()
                                if decoded.startswith("data:"):
                                    payload = decoded.replace("data:", "").strip()
                                    if payload == "[DONE]":
                                        break
                                    try:
                                        d = json.loads(payload)
                                        token = d.get("token") or d.get("text") or ""
                                    except Exception:
                                        token = payload
                                    full += token
                                    answer_box.markdown(f"**Assistant:** {full}")
                            st.session_state.history.append({"user": question, "assistant": full, "ts": time.time()})
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")

with col2:
    st.subheader("Tools")
    st.markdown("### Knowledge Base")
    file_kb = st.file_uploader("Upload to KB", key="kb_uploader", type=["pdf","docx","txt","csv"])
    if file_kb and st.button("Add to KB"):
        files = {"file": (file_kb.name, file_kb.getvalue())}
        try:
            with st.spinner("Uploading to KB..."):
                resp = requests.post(f"{API_PREFIX}/add-to-kb", files=files, timeout=2000)
            if resp.ok:
                st.success("Uploaded to KB")
                st.write(resp.json())
            else:
                try:
                    st.error(resp.json())
                except:
                    st.error(resp.text)
        except requests.exceptions.RequestException as e:
            st.error(f"Upload failed: {e}")

    st.markdown("### Image OCR")
    img = st.file_uploader("OCR Image", type=["png","jpg","jpeg"], key="ocr")
    if img and st.button("Extract Text"):
        files = {"file": (img.name, img.getvalue())}
        try:
            with st.spinner("Extracting..."):
                resp = requests.post(f"{API_PREFIX}/extract-text-from-image", files=files, timeout=2000)
            if resp.ok:
                st.success("Extracted")
                try:
                    data = resp.json()
                    st.text_area("OCR Result", data.get("answer", ""), height=250)
                except Exception:
                    st.text_area("OCR Result", resp.text, height=250)
            else:
                st.error(f"OCR endpoint error: {resp.status_code} {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

    st.markdown("### Audio → Text")
    aud = st.file_uploader("Audio file", type=["wav","mp3","m4a"], key="audio")
    if aud and st.button("Transcribe Audio"):
        files = {"file": (aud.name, aud.getvalue())}
        try:
            with st.spinner("Transcribing..."):
                resp = requests.post(f"{API_PREFIX}/transcribe-audio", files=files, timeout=2000)
            if resp.ok:
                st.success("Transcribed")
                data = resp.json()
                st.text_area("Transcript", data.get("answer", ""), height=250)
            else:
                st.error(f"Transcribe error: {resp.status_code} {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

st.markdown("---")
st.subheader("Conversation History")
if st.session_state.history:
    for m in reversed(st.session_state.history):
        st.markdown(f"**You:** {m['user']}")
        st.markdown(f"**Assistant:** {m['assistant']}")
        st.write("---")
else:
    st.info("No conversation yet.")
