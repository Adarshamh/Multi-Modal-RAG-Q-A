import os
from .config import OLLAMA_TEXT_MODEL, OLLAMA_VISION_MODEL, OLLAMA_LLAVA_MODEL
from .logger import logger

def select_model(mode: str = "text"):
    """
    mode: 'text'|'vision'|'llava'
    """
    mode = mode.lower()
    if mode == "vision":
        logger.info(f"Selecting vision model: {OLLAMA_VISION_MODEL}")
        return OLLAMA_VISION_MODEL
    if mode == "llava":
        logger.info(f"Selecting llava model: {OLLAMA_LLAVA_MODEL}")
        return OLLAMA_LLAVA_MODEL
    logger.info(f"Selecting text model: {OLLAMA_TEXT_MODEL}")
    return OLLAMA_TEXT_MODEL
