import os
from docx import Document
import pandas as pd
import PyPDF2

def extract_text(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    elif ext == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext in [".py", ".js", ".cs"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return None
