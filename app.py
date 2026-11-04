import json
import threading
import uuid
from typing import Iterator, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from configuration import get_params
from data_extraction import DocumentProcessor, UnsupportedFormatError
from logger import logger
from model_call import get_model
from schemas import (
    PendingDoc,
    PendingDocFull,
    PendingDocView,
    QuerySession,
    ReviewResult,
    SearchResponse,
)
from vector_store import get_store
load_dotenv()


app = FastAPI()
processor = DocumentProcessor()

PENDING: dict[str, PendingDoc] = {}
QUERIES: dict[str, QuerySession] = {}
_state_lock = threading.Lock()

PREVIEW_CHARS = 2000

SYSTEM_PROMPT = (
    "You are a Kiwi an AI model developed by Acer Biomedicals LLP, which is a medical equipment manufacturing company located in noida, uttar pradesh , india . Compare documents strictly on the "
    "if someone ask about you your parent company never mention claude haiku or anthropic always say i am Kiwi developed by acer biomdicals located in india."
    "information they contain; never invent specs. Say clearly when data is missing."
)


class ReviewDecision(BaseModel):
    approved: bool
    edited_text: Optional[str] = None  


class CompareRequest(BaseModel):
    query_id: str
    selected: list[str] = Field(min_length=1) 
    instructions: Optional[str] = None


def _read_and_extract(file: UploadFile) -> str:
    content = file.file.read()
    try:
        return processor.extract(file.filename or "", content)
    except UnsupportedFormatError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="extraction failed")


# ---------------- Step 1: upload -> HITL review -> save ----------------
@app.post("/upload", response_model=PendingDocView)
def upload(file: UploadFile = File(...), name: str = Form(...)) -> PendingDocView:
    """Extract karta hai, par save NAHI karta. Human review ka wait karta hai."""
    text = _read_and_extract(file)
    doc_id = uuid.uuid4().hex
    with _state_lock:
        PENDING[doc_id] = {"name": name, "filename": file.filename or "", "text": text}
    return {
        "doc_id": doc_id,
        "name": name,
        "status": "pending_review",
        "chars": len(text),
        "preview": text[:PREVIEW_CHARS],
    }


@app.get("/pending/{doc_id}", response_model=PendingDocFull)
def get_pending(doc_id: str) -> PendingDocFull:
    pending = PENDING.get(doc_id)
    if not pending:
        raise HTTPException(status_code=404, detail="doc_id not found")
    return {"doc_id": doc_id, "name": pending["name"], "text": pending["text"]}


@app.post("/review/{doc_id}", response_model=ReviewResult)
def review(doc_id: str, decision: ReviewDecision) -> ReviewResult:
    """Human approve/reject (aur optionally edit) karta hai."""
    with _state_lock:
        pending = PENDING.pop(doc_id, None)
    if not pending:
        raise HTTPException(status_code=404, detail="doc_id not found")

    if not decision.approved:
        logger.info("document '%s' rejected by reviewer", pending["name"])
        return {"doc_id": doc_id, "status": "rejected", "chunks": 0}

    text = (decision.edited_text or pending["text"]).strip()
    chunks = processor.chunk(text, pending["name"])
    if not chunks:
        raise HTTPException(status_code=422, detail="empty text, save nahi ho sakta")
    try:
        n = get_store().add(pending["name"], text, chunks)
    except Exception:
        logger.exception("save failed")
        raise HTTPException(status_code=500, detail="vector store save failed")
    return {"doc_id": doc_id, "status": "stored", "chunks": n}


@app.get("/documents")
def list_documents() -> dict[str, list[str]]:
    return {"documents": get_store().list_documents()}


# ---------------- Step 2: search -> HITL selection -> streamed compare ----------------
@app.post("/search", response_model=SearchResponse)
def search(file: UploadFile = File(...), k: int = Form(5)) -> SearchResponse:
    text = _read_and_extract(file)
    max_q = get_params()["max_query_chunks"]
    texts = [c["content"] for c in processor.chunk(text, "query")][:max_q]
    try:
        hits = get_store().search(texts, k=k)
    except Exception:
        logger.exception("search failed")
        raise HTTPException(status_code=500, detail="similarity search failed")

    query_id = uuid.uuid4().hex
    with _state_lock:
        QUERIES[query_id] = {"text": text, "hits": hits}
    return {"query_id": query_id, "status": "awaiting_selection", "candidates": hits}


def _sse(event: str, payload: object) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/compare")
def compare(req: CompareRequest) -> StreamingResponse:
    session = QUERIES.get(req.query_id)
    if not session:
        raise HTTPException(status_code=404, detail="query_id not found")

    store = get_store()
    unknown = [n for n in req.selected if not store.has(n)]
    if unknown:
        raise HTTPException(status_code=404, detail=f"unknown documents: {unknown}")

    cap = get_params()["max_doc_chars"]
    candidates = "\n\n".join(
        f'<candidate name="{n}">\n{store.get_text(n)[:cap]}\n</candidate>' for n in req.selected
    )
    prompt = (
        f"<main_file>\n{session['text'][:cap]}\n</main_file>\n\n"
        f"<candidates>\n{candidates}\n</candidates>\n\n"
        "Batao ki main file ke hisaab se kaun si candidate files ke specs sabse accurate hain, "
        "aur sabka ek comparison table banao. Har claim ke saath file ka naam do."
    )
    if req.instructions:
        prompt += f"\n\nExtra instructions: {req.instructions}"

    def event_stream() -> Iterator[str]:
        try:
            for token in get_model().stream(prompt, system=SYSTEM_PROMPT):
                yield _sse("token", token)
            yield _sse("done", "[DONE]")
        except Exception:
            logger.exception("compare stream failed")
            yield _sse("error", "model call failed")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}