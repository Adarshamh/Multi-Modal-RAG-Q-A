import logging
from .embedding_manager import retrieve_from_kb
try:
    from ollama import Ollama
    ollama_available = True
    ollama_client = Ollama()
except:
    ollama_available = False
    import openai

logging.basicConfig(filename="logs/app.log", level=logging.INFO)

def chat_with_llm(messages: list):
    try:
        if ollama_available:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            response = ollama_client.completion(prompt=prompt, model="llama3-8b")
            return response.text
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return response.choices[0].message.content
    except Exception as e:
        logging.error(f"LLM Error: {str(e)}")
        return "⚠️ LLM Error. Check logs."
