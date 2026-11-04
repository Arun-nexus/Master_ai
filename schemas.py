from typing_extensions import Literal, TypedDict


class Params(TypedDict):
    model: str
    max_tokens: int
    embedding_model: str
    storage_path: str
    chunk_size: int
    chunk_overlap: int
    max_query_chunks: int
    max_doc_chars: int


class ChunkMeta(TypedDict):
    name: str
    chunk_id: int


class Chunk(TypedDict):
    content: str
    metadata: ChunkMeta


class SearchHit(TypedDict):
    name: str
    score: float
    snippet: str


# ---- HITL state (server memory) ----
class PendingDoc(TypedDict):
    name: str
    filename: str
    text: str


class QuerySession(TypedDict):
    text: str
    hits: list[SearchHit]


# ---- API responses ----
class PendingDocView(TypedDict):
    doc_id: str
    name: str
    status: Literal["pending_review"]
    chars: int
    preview: str


class PendingDocFull(TypedDict):
    doc_id: str
    name: str
    text: str


class ReviewResult(TypedDict):
    doc_id: str
    status: Literal["stored", "rejected"]
    chunks: int


class SearchResponse(TypedDict):
    query_id: str
    status: Literal["awaiting_selection"]
    candidates: list[SearchHit]