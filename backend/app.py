import os
import logging
import sqlite3
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import json
import datetime

# Import routers using relative imports
from .routes import (
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

from .core.config import LOG_PATH, DB_PATH

# prepare logger and directories
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)

# initialize simple sqlite DB for analytics & sessions (idempotent)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
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
c.execute("""
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at TEXT
)
""")
conn.commit()
conn.close()

app = FastAPI(title="Multi-Modal-RAG-Q-A Backend (Ollama offline/online)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers under /api
app.include_router(file_chat.router, prefix="/api")
app.include_router(url_chat.router, prefix="/api")
app.include_router(ocr_image.router, prefix="/api")
app.include_router(transcribe_audio.router, prefix="/api")
app.include_router(knowledge_base.router, prefix="/api")
app.include_router(summarize.router, prefix="/api")
app.include_router(chat_stream.router, prefix="/api")
app.include_router(retriever_routes.router, prefix="/api")
app.include_router(inference_routes.router, prefix="/api")

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/analytics/log")
def log_entry(payload: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO analytics (session_id,user_id,message,response,latency,tokens,created_at) VALUES (?,?,?,?,?,?,?)",
            (
                payload.get("session_id"),
                payload.get("user_id"),
                payload.get("message"),
                payload.get("response"),
                payload.get("latency"),
                payload.get("tokens"),
                datetime.datetime.utcnow().isoformat()
            )
        )
        conn.commit()
        conn.close()
        return {"ok": True}
    except Exception as e:
        logging.exception("analytics log error")
        return Response(
            content=json.dumps({"ok": False, "error": str(e)}),
            media_type="application/json",
            status_code=500
        )

@app.get("/api/analytics/stats")
def stats():
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT created_at, user_id, message, latency, tokens FROM analytics ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()
    timeline = [
        {"timestamp": r[0], "user_id": r[1], "message": r[2], "latency": r[3], "tokens": r[4]} for r in rows
    ]
    return {"timeline": timeline}
