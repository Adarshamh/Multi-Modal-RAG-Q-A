from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from backend.core.model_selector import select_model
from backend.core.logger import logger
import ollama, tempfile, os, base64

router = APIRouter()

@router.post("/infer")
async def infer_route(input_type: str = Form(...), prompt: str = Form(...), file: UploadFile = File(None)):
    try:
        model = select_model(input_type)
        # if file present, save temp and optionally include base64 (models vary on how to accept images)
        if file:
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(await file.read())
            tmp.flush()
            tmp.close()
            # Some multimodal models require base64 embedding; put it in payload if needed
            with open(tmp.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            try:
                os.unlink(tmp.name)
            except:
                pass
            # Send prompt + (optionally) b64 as part of message; adjust per model expectations
            message = {"role":"user","content": f"{prompt}\n\n[[image_b64:{b64[:100]}...]]"}
        else:
            message = {"role":"user","content": prompt}
        resp = ollama.chat(model=model, messages=[message])
        if isinstance(resp, dict):
            ans = resp.get("text") or (resp.get("choices")[0]["message"]["content"] if resp.get("choices") else None)
        else:
            ans = str(resp)
        return {"answer": ans}
    except Exception as e:
        logger.exception("infer_route error")
        raise HTTPException(status_code=500, detail=str(e))
