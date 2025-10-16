from backend.core.config import OLLAMA_TEXT_MODEL, OLLAMA_VISION_MODEL, OLLAMA_LLAVA_MODEL

def select_model(input_type: str):
    t = (input_type or "text").lower()
    if t in ["image", "vision", "img"]:
        return OLLAMA_VISION_MODEL
    if t in ["multimodal", "llava", "image+text", "video", "audio"]:
        return OLLAMA_LLAVA_MODEL
    return OLLAMA_TEXT_MODEL
