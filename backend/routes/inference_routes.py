from fastapi import APIRouter, Form
from ..core.logger import logger

router = APIRouter()

@router.post("/infer")
def infer(prompt: str = Form(...)):
    """
    Simple synchronous inference placeholder (non-streaming).
    """
    logger.info("infer called")
    return {"ok": False, "message": "Infer route not implemented (future enhancement)."}
