import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.logger import logger

# import routers
from backend.routes import (
    file_chat,
    url_chat,
    ocr_image,
    summarize,
    transcribe_audio,
    knowledge_base,
    chat_stream,
    retriever_routes,
    inference_routes
)

app = FastAPI(title="MM-RAG Multi-Modal RAG Backend")

# CORS (allow all for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
app.include_router(file_chat.router)
app.include_router(url_chat.router)
app.include_router(ocr_image.router)
app.include_router(summarize.router)
app.include_router(transcribe_audio.router)
app.include_router(knowledge_base.router)
app.include_router(chat_stream.router)
app.include_router(retriever_routes.router)
app.include_router(inference_routes.router)

@app.get("/")
def root():
    logger.info("Root endpoint called")
    return {"message": "MM-RAG backend running"}
