import os
from io import StringIO
from ..core.logger import logger
from PyPDF2 import PdfReader
import docx
import pandas as pd

def extract_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(path)
            text = []
            for p in reader.pages:
                text.append(p.extract_text() or "")
            return "\n".join(text)
        elif ext == ".docx":
            doc = docx.Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext in [".txt", ".md", ".py", ".js"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".csv":
            df = pd.read_csv(path, encoding="utf-8", errors="ignore")
            return df.to_csv(index=False)
        else:
            logger.warning("Unsupported extract type: %s", ext)
            return ""
    except Exception as e:
        logger.exception("Failed to extract text from %s", path)
        return ""
