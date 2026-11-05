import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
from data_extraction import DocumentProcessor
from vector_store import get_store

load_dotenv()

base = Path("Data/faiss")
docs = json.loads((base / "documents.json").read_text(encoding="utf-8"))["docs"]
proc = DocumentProcessor()
store = get_store()
for name in docs:
    p = base / "texts" / f"{hashlib.sha1(name.encode('utf-8')).hexdigest()}.txt"
    text = p.read_text(encoding="utf-8")
    store.add(name, text, proc.chunk(text, name))
    print("done", name)