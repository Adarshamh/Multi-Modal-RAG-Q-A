from fastapi import APIRouter
from ..core.rag_engine import retrieve
from ..core.logger import logger

router = APIRouter()

@router.get("/search")
def search(q: str, k: int = 5):
    try:
        results = retrieve(q, k=k)
        return {"ok": True, "results": results}
    except Exception as e:
        logger.exception("retriever search error")
        return {"ok": False, "error": str(e)}
