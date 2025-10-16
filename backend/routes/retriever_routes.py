from fastapi import APIRouter, Query
from backend.core.rag_engine import retrieve
from backend.core.logger import logger

router = APIRouter(prefix="/api/retriever", tags=["retriever"])

@router.get("/query")
def query_docs(q: str = Query(...)):
    try:
        results = retrieve(q)
        return {"results": results}
    except Exception as e:
        logger.exception("retriever query failed")
        return {"error": str(e)}
