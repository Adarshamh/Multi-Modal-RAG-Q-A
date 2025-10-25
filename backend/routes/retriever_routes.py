from fastapi import APIRouter, Form
from ..core.embedding_manager import get_manager

router = APIRouter()

@router.post("/search")
def search(query: str = Form(...), k: int = Form(5)):
    manager = get_manager()
    results = manager.hybrid_search(query, k=k)
    return {"results": results}
