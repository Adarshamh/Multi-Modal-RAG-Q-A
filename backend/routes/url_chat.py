from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from core.rag_engine import chat_with_llm
from core.embedding_manager import retrieve_from_kb
from langchain_community.document_loaders import WebBaseLoader
import os

router = APIRouter()

class URLQuery(BaseModel):
    url: str
    question: str
    template: str = "qa"
    session_id: str = None

prompt_templates = {
    "qa": "Answer the user's question based on the content:\n{content}\nQuestion: {question}",
    "summary": "Summarize the following content in bullet points:\n{content}",
    "code_review": "Review the following code and provide suggestions:\n{content}"
}

def apply_template(template_name, content, question=""):
    template = prompt_templates.get(template_name, "{content}\nQuestion: {question}")
    return template.format(content=content, question=question)

session_memory = {}
def get_session_messages(session_id):
    return session_memory.get(session_id, [])

def append_session_message(session_id, role, content):
    session_memory.setdefault(session_id, []).append({"role": role, "content": content})

@router.post("/chat-with-url")
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
        return JSONResponse(status_code=500, content={"error": str(e)})
