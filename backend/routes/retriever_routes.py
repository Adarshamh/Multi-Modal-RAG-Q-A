from fastapi import APIRouter
from ..core.rag_engine import retrieve
from ..core.logger import logger

router = APIRouter()

@router.get("/retrieve")
def retrieve_endpoint(q: str, k: int = 5):
    try:
        docs = retrieve(q, k=k)
        snippets = [{"text": d.page_content, "meta": d.metadata} for d in docs]
        return {"results": snippets}
    except Exception as e:
        logger.exception("retrieve endpoint error")
        return {"error": str(e)}
