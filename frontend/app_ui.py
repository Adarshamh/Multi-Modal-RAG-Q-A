import streamlit as st
import requests
import json
import time
import os
from sseclient import SSEClient

# ----------------------------
# Backend configuration
# ----------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="🤖 Multi-Modal RAG Assistant", layout="wide")
st.title("🤖 Multi-Modal RAG Assistant")
st.caption("Ask questions from your documents, images, or videos using local AI (Ollama)")

# ----------------------------
# Session state
# ----------------------------
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Tabs: Chat and OCR
# ----------------------------
tab_chat, tab_ocr = st.tabs(["💬 Ask Question / File Chat", "🖼️ Extract Text from Image"])

# ----------------------------
# Chat / Knowledge Base tab
# ----------------------------
with tab_chat:
    st.subheader("Ask a Question or Add Document to Knowledge Base")

    # File upload (optional)
    uploaded_file = st.file_uploader(
        "Upload a document or media (optional). Any question entered will be about this file if uploaded.",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "mp4", "wav", "csv"],
        key="file_upload"
    )

    # Add to Knowledge Base button
    if uploaded_file and st.button("📤 Add to Knowledge Base"):
        with st.spinner(f"Uploading {uploaded_file.name} to Knowledge Base..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                resp = requests.post(f"{BACKEND_URL}/add-to-kb", files=files, timeout=120)
                if resp.ok:
                    st.success(f"✅ {uploaded_file.name} added to Knowledge Base")
                else:
                    st.error(f"❌ Upload failed: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Upload failed: {str(e)}")

    # User question input
    question = st.text_area("Enter your question here", placeholder="e.g., Summarize the uploaded document or ask any question")

    # Ask button
    if st.button("🚀 Ask") and question.strip():
        with st.spinner("Generating answer..."):
            st.markdown("### 🧩 Answer")
            answer_box = st.empty()
            full_answer = ""

            try:
                # Decide endpoint based on file upload
                if uploaded_file:
                    # File chat
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    data = {"question": question, "template": "qa", "session_id": st.session_state.session_id}
                    resp = requests.post(f"{BACKEND_URL}/file-chat", files=files, data=data, timeout=300)
                    if resp.ok:
                        full_answer = resp.json().get("answer", "")
                    else:
                        st.error(f"❌ File chat failed: {resp.text}")
                        full_answer = ""
                else:
                    # General question (streamed)
                    with requests.post(
                        f"{BACKEND_URL}/chat-stream",
                        json={"question": question},
                        stream=True,
                        timeout=800,
                    ) as r:
                        if not r.ok:
                            st.error(f"Backend returned {r.status_code}: {r.text}")
                        else:
                            client = SSEClient(r)
                            for event in client.events():
                                if not event.data:
                                    continue
                                if event.data == "[DONE]":
                                    break
                                try:
                                    data = json.loads(event.data)
                                    token = data.get("token", "")
                                except:
                                    token = event.data
                                full_answer += token
                                answer_box.markdown(full_answer)

                # Save history
                if full_answer:
                    st.session_state.history.append({"user": question, "assistant": full_answer, "ts": time.time()})

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection error: {str(e)}")

    # Conversation history
    st.markdown("---")
    st.subheader("Conversation History")
    if st.session_state.history:
        for m in reversed(st.session_state.history):
            st.markdown(f"**You:** {m['user']}")
            st.markdown(f"**Assistant:** {m['assistant']}")
            st.write("---")
    else:
        st.info("No conversation yet.")

# ----------------------------
# OCR Tab
# ----------------------------
with tab_ocr:
    st.subheader("Extract Text from Image")
    image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="ocr_file")
    if image_file and st.button("📖 Extract Text"):
        with st.spinner("Extracting text using OCR..."):
            try:
                files = {"file": (image_file.name, image_file.getvalue())}
                resp = requests.post(f"{BACKEND_URL}/extract-text", files=files, timeout=120)
                if resp.ok:
                    text = resp.json().get("text", "")
                    st.text_area("Extracted Text", text, height=300)
                else:
                    st.error(f"❌ OCR failed: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ OCR failed: {str(e)}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Powered by 🧠 Multi-Modal RAG • Local AI • Streamlit Frontend")
