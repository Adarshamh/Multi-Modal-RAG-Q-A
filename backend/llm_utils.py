import os
from ollama import Ollama

ollama_client = Ollama(model_name="llama3-8b-8192")

def ask_llm(session_memory, question, context=""):
    prompt = f"{context}\n\nQuestion: {question}"
    messages = [{"role": m["role"], "content": m["content"]} for m in session_memory] + [{"role": "user", "content": prompt}]
    response = ollama_client.chat(messages)
    return response["content"]
