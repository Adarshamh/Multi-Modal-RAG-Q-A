from .config import OLLAMA_TEXT_MODEL, OLLAMA_VISION_MODEL, OLLAMA_EMBED_MODEL

def select_model(modality="text"):
    """
    Simple model selector. Returns model name string configured in .env
    """
    if modality == "vision":
        return OLLAMA_VISION_MODEL
    if modality == "embed":
        return OLLAMA_EMBED_MODEL
    return OLLAMA_TEXT_MODEL
