import gradio as gr
import requests
import uuid

API_URL = "http://127.0.0.1:8000"
session_id = str(uuid.uuid4())

# ------------------- Helper Functions -------------------
def chat_with_file(file, question, template="qa"):
    if file is None or question.strip() == "":
        return "⚠️ Please provide both a file and a question."
    files = {"file": open(file.name, "rb")}
    data = {"question": question, "template": template, "session_id": session_id}
    try:
        r = requests.post(f"{API_URL}/chat-with-file", files=files, data=data)
        return r.json().get("answer", "⚠️ No answer returned.")
    except Exception as e:
        return f"❌ Error: {str(e)}"

def chat_with_url(url, question, template="qa"):
    if url.strip() == "" or question.strip() == "":
        return "⚠️ Please provide both a URL and a question."
    data = {"url": url, "question": question, "template": template, "session_id": session_id}
    try:
        r = requests.post(f"{API_URL}/chat-with-url", json=data)
        return r.json().get("answer", "⚠️ No answer returned.")
    except Exception as e:
        return f"❌ Error: {str(e)}"

def extract_text_from_image(file):
    if file is None:
        return "⚠️ Please provide an image file."
    files = {"file": open(file.name, "rb")}
    try:
        r = requests.post(f"{API_URL}/extract-text-from-image", files=files)
        return r.json().get("answer", "⚠️ No text extracted.")
    except Exception as e:
        return f"❌ Error: {str(e)}"

def transcribe_audio(file):
    if file is None:
        return "⚠️ Please provide an audio file."
    files = {"file": open(file.name, "rb")}
    try:
        r = requests.post(f"{API_URL}/transcribe-audio", files=files)
        return r.json().get("answer", "⚠️ No transcript returned.")
    except Exception as e:
        return f"❌ Error: {str(e)}"

def add_to_kb(file):
    if file is None:
        return "⚠️ Please provide a file to add to the knowledge base."
    files = {"file": open(file.name, "rb")}
    try:
        r = requests.post(f"{API_URL}/add-to-kb", files=files)
        return r.json().get("message", "⚠️ Could not add to KB.")
    except Exception as e:
        return f"❌ Error: {str(e)}"

def auto_summarize(file):
    if file is None:
        return "⚠️ Please provide a file to summarize."
    files = {"file": open(file.name, "rb")}
    data = {"question": "Summarize this document in bullet points.", "template": "summary", "session_id": session_id}
    try:
        r = requests.post(f"{API_URL}/chat-with-file", files=files, data=data)
        return r.json().get("answer", "⚠️ No summary returned.")
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ------------------- GRADIO UI -------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🌐 Multi-Modal RAG Q&A – Multi-Modal AI Assistant")
    gr.Markdown("Upload files, URLs, images, or audio and ask questions. Left panel: Inputs | Right panel: Answer")

    with gr.Row():
        # ------------------- LEFT COLUMN (Inputs) -------------------
        with gr.Column(scale=1):
            with gr.Tab("Chat with File"):
                file_input = gr.File(file_types=[".txt",".docx",".pdf",".csv",".py",".js",".cs"])
                file_question = gr.Textbox(label="Your Question", placeholder="Ask something about the file...")
                file_template = gr.Dropdown(label="Template", choices=["qa", "summary", "code_review"], value="qa")
                file_btn = gr.Button("Ask")

            with gr.Tab("Chat with URL"):
                url_input = gr.Textbox(label="Website URL")
                url_question = gr.Textbox(label="Your Question")
                url_template = gr.Dropdown(label="Template", choices=["qa", "summary", "code_review"], value="qa")
                url_btn = gr.Button("Ask")

            with gr.Tab("Extract Text from Image"):
                image_input = gr.Image(type="filepath")
                image_btn = gr.Button("Extract Text")

            with gr.Tab("Transcribe Audio"):
                audio_input = gr.Audio(type="filepath")
                audio_btn = gr.Button("Transcribe")

            with gr.Tab("Auto-Summarization"):
                summary_file = gr.File(file_types=[".txt",".docx",".pdf",".csv",".py",".js",".cs"])
                summary_btn = gr.Button("Summarize")

            with gr.Tab("Knowledge Base"):
                kb_file = gr.File(file_types=[".txt",".docx",".pdf",".csv",".py",".js",".cs"])
                kb_btn = gr.Button("Add to KB")

        # ------------------- RIGHT COLUMN (Answer) -------------------
        with gr.Column(scale=3):
            answer_output = gr.Markdown(value="### Answers will appear here...", elem_classes="answer-box")

    # ------------------- BUTTON CLICKS -------------------
    file_btn.click(chat_with_file, inputs=[file_input, file_question, file_template], outputs=answer_output)
    url_btn.click(chat_with_url, inputs=[url_input, url_question, url_template], outputs=answer_output)
    image_btn.click(extract_text_from_image, inputs=image_input, outputs=answer_output)
    audio_btn.click(transcribe_audio, inputs=audio_input, outputs=answer_output)
    summary_btn.click(auto_summarize, inputs=summary_file, outputs=answer_output)
    kb_btn.click(add_to_kb, inputs=kb_file, outputs=answer_output)

demo.launch()
