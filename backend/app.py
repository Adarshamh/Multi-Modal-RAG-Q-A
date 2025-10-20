import os 
import logging
import sqlite3
import json
import datetime
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from backend.routes import (
    file_chat,
    url_chat,
    ocr_image,
    transcribe_audio,
    knowledge_base,
    summarize,
    chat_stream,
    retriever_routes,
    inference_routes,
)

# Import configuration
from backend.core.config import LOG_DIR, DB_PATH, LOG_FILE
from backend.core.logger import logger

# --------------------------------------------------------------------
# Directory setup
# --------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# --------------------------------------------------------------------
# SQLite database initialization (for analytics & sessions)
# --------------------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Analytics table
c.execute("""
CREATE TABLE IF NOT EXISTS analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    user_id TEXT,
    message TEXT,
    response TEXT,
    latency REAL,
    tokens INTEGER,
    created_at TEXT
)
""")

# Sessions table
c.execute("""
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at TEXT
)
""")

conn.commit()
conn.close()

# --------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger.info("Logger initialized, logs will be saved to %s", LOG_FILE)

# --------------------------------------------------------------------
# FastAPI initialization
# --------------------------------------------------------------------
app = FastAPI(
    title="Multi-Modal-RAG-Q-A Backend (Ollama offline/online)",
    version="1.0.0",
    description="Handles multimodal (text, audio, image, URL) inputs and RAG responses using Ollama + FAISS."
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Register routes
# --------------------------------------------------------------------
app.include_router(file_chat.router, prefix="/api")
app.include_router(url_chat.router, prefix="/api")
app.include_router(ocr_image.router, prefix="/api")
app.include_router(transcribe_audio.router, prefix="/api")
app.include_router(knowledge_base.router, prefix="/api")
app.include_router(summarize.router, prefix="/api")
app.include_router(chat_stream.router, prefix="/api")
app.include_router(retriever_routes.router, prefix="/api")
app.include_router(inference_routes.router, prefix="/api")

# --------------------------------------------------------------------
# Health check endpoint
# --------------------------------------------------------------------
@app.get("/api/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Backend active and healthy"}

# --------------------------------------------------------------------
# Analytics endpoints
# --------------------------------------------------------------------
@app.post("/api/analytics/log")
def log_entry(payload: dict):
    """Log user interaction and model response."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO analytics (session_id, user_id, message, response, latency, tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            payload.get("session_id"),
            payload.get("user_id"),
            payload.get("message"),
            payload.get("response"),
            payload.get("latency"),
            payload.get("tokens"),
            datetime.datetime.utcnow().isoformat()
        ))
        conn.commit()
        conn.close()
        return {"ok": True}
    except Exception as e:
        logger.exception("Analytics log error")
        return Response(
            content=json.dumps({"ok": False, "error": str(e)}),
            media_type="application/json",
            status_code=500,
        )

@app.get("/api/analytics/stats")
def stats():
    """Return aggregated analytics stats."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*), AVG(latency) FROM analytics")
    row = c.fetchone()
    conn.close()
    total = row[0] or 0
    avg_latency = float(row[1]) if row[1] else 0.0
    return {"requests": total, "avg_latency": avg_latency}

@app.get("/api/analytics/timeline")
def timeline(limit: int = 50):
    """Return recent analytics timeline."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT created_at, user_id, message, latency, tokens
        FROM analytics
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    timeline = [
        {
            "timestamp": r[0],
            "user_id": r[1],
            "message": r[2],
            "latency": r[3],
            "tokens": r[4],
        }
        for r in rows
    ]
    return {"timeline": timeline}
