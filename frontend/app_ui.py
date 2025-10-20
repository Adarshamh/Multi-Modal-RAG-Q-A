# frontend/app_ui.py
import streamlit as st
import requests
import os
import json
import time
from streamlit_mic_recorder import mic_recorder

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

    st.markdown("### Audio → Text (Single-shot)")
    st.markdown("Record a short clip and transcribe it using Whisper on the backend.")
    # mic_recorder returns None or a dict { 'blob':..., 'bytes':..., 'mime':... } depending on version
    audio = mic_recorder(start_prompt="🎤 Record", stop_prompt="🛑 Stop")
    if audio:
        # older versions return dict with key 'bytes' or 'blob'; try to handle both
        audio_bytes = None
        if isinstance(audio, dict):
            audio_bytes = audio.get("bytes") or audio.get("blob")
        else:
            audio_bytes = audio
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            col_a, col_b = st.columns([1,2])
            with col_a:
                add_to_kb = st.checkbox("Add transcript to KB", value=False, key="add_kb_single")
            with col_b:
                lang_input = st.text_input("Language (optional, e.g. 'en')", value="", key="lang_single")
            if st.button("Transcribe Recording"):
                with st.spinner("Sending audio to backend..."):
                    files = {"file": ("recording.wav", audio_bytes, "audio/wav")}
                    data = {"add_to_kb": str(add_to_kb).lower(), "filename": f"mic_{st.session_state.session_id}.wav", "language": lang_input}
                    try:
                        resp = requests.post(f"{API_PREFIX}/transcribe-audio", files=files, data=data, timeout=600)
                        if resp.ok:
                            out = resp.json()
                            st.success("Transcription complete")
                            st.text_area("Transcript", out.get("answer", ""), height=300)
                            if out.get("answer"):
                                st.session_state.history.append({"user": "Audio (single-shot)", "assistant": out.get("answer"), "ts": time.time()})
                        else:
                            st.error(f"Transcription failed: {resp.status_code} {resp.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection error: {e}")

    st.markdown("### Live Captions (Chunked) — emulated continuous transcription")
    st.markdown("Start a session and repeatedly record short chunks (e.g., 3–6s). Each chunk is sent to backend and appended to captions.")
    if "live_captions" not in st.session_state:
        st.session_state.live_captions = ""
        st.session_state.live_session = False
    if st.button("Start Live Session") and not st.session_state.live_session:
        st.session_state.live_session = True
        st.session_state.live_captions = ""
        st.success("Live session started — press 'Record Chunk' repeatedly.")
    if st.button("Stop Live Session") and st.session_state.live_session:
        st.session_state.live_session = False
        st.info("Live session stopped.")
    if st.session_state.live_session:
        st.markdown("**Live captions (appending):**")
        st.text_area("Live Captions", value=st.session_state.live_captions, height=200, key="live_captions_area", disabled=True)
        # Record one chunk
        chunk = mic_recorder(start_prompt="🔴 Record Chunk", stop_prompt="⏹ Stop Chunk")
        if chunk:
            chunk_bytes = None
            if isinstance(chunk, dict):
                chunk_bytes = chunk.get("bytes") or chunk.get("blob")
            else:
                chunk_bytes = chunk
            if chunk_bytes:
                st.audio(chunk_bytes, format="audio/wav")
                lang = st.text_input("Language for partials (optional)", value="", key="lang_partial")
                with st.spinner("Sending chunk to backend..."):
                    try:
                        files = {"file": ("chunk.wav", chunk_bytes, "audio/wav")}
                        resp = requests.post(f"{API_PREFIX}/transcribe-partial", files=files, data={"language": lang}, timeout=120)
                        if resp.ok:
                            part = resp.json().get("partial", "")
                            if part:
                                st.session_state.live_captions += (" " + part).strip()
                        else:
                            st.error(f"Partial transcription failed: {resp.status_code}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection error: {e}")
    st.markdown("---")

st.markdown("---")
st.subheader("Conversation History")
if st.session_state.history:
    for m in reversed(st.session_state.history):
        st.markdown(f"**You:** {m['user']}")
        st.markdown(f"**Assistant:** {m['assistant']}")
        st.write("---")
else:
    st.info("No conversation yet.")
