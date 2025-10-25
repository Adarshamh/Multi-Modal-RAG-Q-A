import streamlit as st
import requests
import os
import json
import time
import base64
import uuid

st.set_page_config(page_title="Multi-Modal RAG Q&A", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_PREFIX = f"{BACKEND_URL}/api"

st.title("🤖 Multi-Modal RAG Assistant")
st.caption("Ask questions from your documents, images, audio or URLs (Local Ollama)")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("💬 Chat")
    kb_files = st.file_uploader("Upload a document to KB (optional)", type=["pdf","docx","txt","csv"], key="kb_up")
    if kb_files and st.button("Upload to KB"):
        files = {"file": (kb_files.name, kb_files.getvalue())}
        with st.spinner("Uploading to KB..."):
            resp = requests.post(f"{API_PREFIX}/add-to-kb", files=files)
        if resp.ok:
            st.success("Uploaded to KB")
            try:
                st.json(resp.json())
            except:
                st.write(resp.text)
        else:
            st.error(resp.text)

    question = st.text_area("Ask a question (about KB or general)", height=140)
    ask = st.button("Ask")

    if ask:
        if not question:
            st.info("Please type a question.")
        else:
            # if user uploaded, call file-chat streaming — else chat-stream
            if kb_files:
                files = {"file": (kb_files.name, kb_files.getvalue())}
                data = {"question": question, "session_id": st.session_state.session_id}
                try:
                    with st.spinner("Uploading & asking..."):
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
                                answer_box.markdown(f"**Assistant:** {full}")
                        st.session_state.history.append({"user": question, "assistant": full, "ts": time.time()})
                    else:
                        st.error(f"Backend error: {resp.status_code} {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
            else:
                try:
                    with st.spinner("Contacting model..."):
                        url = f"{API_PREFIX}/chat-stream"
                        with requests.post(url, json={"question": question}, stream=True, timeout=300) as r:
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
                                        payload = decoded[len("data:"):].strip()
                                        if payload == "[DONE]":
                                            break
                                        try:
                                            d = json.loads(payload)
                                            token = d.get("token") or d.get("content") or ""
                                        except:
                                            token = payload
                                        full += token
                                        answer_box.markdown(f"**Assistant:** {full}")
                                st.session_state.history.append({"user": question, "assistant": full, "ts": time.time()})
                except Exception as e:
                    st.error(f"Connection error: {e}")

with col2:
    st.subheader("Tools")
    st.markdown("### OCR (Image → Text)")
    img = st.file_uploader("Upload image for OCR", type=["png","jpg","jpeg"], key="ocr")
    if img and st.button("Extract Text from Image"):
        files = {"file": (img.name, img.getvalue())}
        with st.spinner("Extracting..."):
            resp = requests.post(f"{API_PREFIX}/extract-text-from-image", files=files, timeout=120)
        if resp.ok:
            data = resp.json()
            st.success("Extracted")
            st.text_area("OCR Result", data.get("answer", ""), height=250)
        else:
            st.error(f"OCR error: {resp.status_code} {resp.text}")

    st.markdown("### Audio → Text (Single-shot)")
    aud = st.file_uploader("Upload audio", type=["wav","mp3","m4a"], key="audio")
    if aud and st.button("Transcribe Audio"):
        files = {"file": (aud.name, aud.getvalue())}
        with st.spinner("Transcribing..."):
            resp = requests.post(f"{API_PREFIX}/transcribe-audio", files=files, timeout=300)
        if resp.ok:
            data = resp.json()
            st.success("Transcribed")
            st.text_area("Transcript", data.get("answer", ""), height=250)
        else:
            st.error(f"Transcribe error: {resp.status_code} {resp.text}")

    st.markdown("### Audio → Text (Live / chunked)")
    st.write("Use this to capture short live audio chunks (client-side recording required). If your browser cannot record, upload audio and click Transcribe.")
    # Try to use the streamlit-mic-recorder component if available
    try:
        from streamlit_mic_recorder import audio_recorder
        rec = audio_recorder()
        if rec and st.button("Send Live Clip (final)"):
            # rec is bytes: send base64 chunk with final=True
            b64 = base64.b64encode(rec).decode("utf-8")
            sess = st.session_state.session_id
            with st.spinner("Sending..."):
                resp = requests.post(f"{API_PREFIX}/transcribe-stream", data={"session_id": sess, "chunk_b64": b64, "final": True})
            if resp.ok:
                data = resp.json()
                st.success("Transcript")
                st.text_area("Live Transcript", data.get("answer", ""), height=200)
            else:
                st.error(resp.text)
    except Exception:
        st.info("Install streamlit-mic-recorder to use live recording, or use the single-shot uploader above.")

st.markdown("---")
st.subheader("Conversation History")
if st.session_state.history:
    for m in reversed(st.session_state.history):
        st.markdown(f"**You:** {m['user']}")
        st.markdown(f"**Assistant:** {m['assistant']}")
        st.write("---")
else:
    st.info("No conversation yet.")
