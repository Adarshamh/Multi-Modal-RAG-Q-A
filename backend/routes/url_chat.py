from fastapi import APIRouter, Form
from ..core.logger import logger

router = APIRouter()

@router.post("/url-chat")
def url_chat(url: str = Form(...), question: str = Form(...)):
    """
    Placeholder for URL ingestion + chat. (Future enhancement)
    """
    logger.info("url_chat called for %s", url)
    return {"ok": False, "message": "url_chat not implemented yet. (Future enhancement)"}
