from fastapi import WebSocket

active_sessions = {}  # session_id -> list of websockets

async def broadcast(session_id, message):
    if session_id in active_sessions:
        for ws in active_sessions[session_id]:
            await ws.send_text(message)
