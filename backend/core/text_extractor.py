import os
from docx import Document
import PyPDF2
from .logger import logger

def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = []
                for p in reader.pages:
                    try:
                        pages.append(p.extract_text() or "")
                    except Exception:
                        pages.append("")
                text = "\n".join(pages)
        elif ext == ".docx":
            doc = Document(path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext in [".txt", ".csv", ".md"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            logger.warning("Unsupported ext for text extraction: %s", ext)
    except Exception as e:
        logger.exception("Failed to extract text from %s", path)
    return text
