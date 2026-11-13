import os
import re
import time
import uuid
from functools import lru_cache
from threading import RLock
from typing import Any

import voyageai
from qdrant_client import QdrantClient, models

from logger import logger
from schemas import Chunk, SearchHit

SRC_RE = re.compile(r"(?m)^## \[(catalogue|manual)\] (.+)$")
CHUNKS = "chunks"
DOCS = "docs"
BATCH = 64
EMBED_BATCH = 64
MODEL = os.getenv("VOYAGE_MODEL", "voyage-4")
DIM = int(os.getenv("EMBED_DIM", "1024"))


@lru_cache(maxsize=1)
def get_voyage() -> voyageai.Client:
    return voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"], max_retries=2, timeout=60)


def embed(texts: list[str], input_type: str) -> list[list[float]]:
    client = get_voyage()
    out: list[list[float]] = []
    for s in range(0, len(texts), EMBED_BATCH):
        part = texts[s : s + EMBED_BATCH]
        for attempt in range(6):
            try:
                res = client.embed(
                    part,
                    model=MODEL,
                    input_type=input_type,
                    output_dimension=DIM,
                    truncation=True,
                )
                out.extend(res.embeddings)
                break
            except voyageai.error.RateLimitError:
                if attempt == 5:
                    raise
                wait = 5 * (attempt + 1)
                logger.warning("voyage rate limit, retrying in %ss", wait)
                time.sleep(wait)
    return out


def _doc_id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{name}"))


def _chunk_id(name: str, i: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:{name}#{i}"))


class QdrantVectorStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self.client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=60,
        )
        self._init_collections()

    def _init_collections(self) -> None:
        if not self.client.collection_exists(CHUNKS):
            self.client.create_collection(
                CHUNKS,
                vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
            )
        self.client.create_payload_index(CHUNKS, "name", models.PayloadSchemaType.KEYWORD)
        if not self.client.collection_exists(DOCS):
            self.client.create_collection(
                DOCS,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
            )

    @staticmethod
    def _info(text: str) -> dict[str, Any]:
        return {"chars": len(text), "sources": [{"kind": k, "file": f} for k, f in SRC_RE.findall(text)]}

    def _remove(self, name: str) -> None:
        self.client.delete(
            CHUNKS,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="name", match=models.MatchValue(value=name))]
                )
            ),
        )
        self.client.delete(DOCS, points_selector=models.PointIdsList(points=[_doc_id(name)]))

    def remove(self, name: str) -> None:
        with self._lock:
            self._remove(name)

    def add(self, name: str, text: str, chunks: list[Chunk]) -> int:
        if not chunks:
            raise ValueError("Chunks list empty hai")
        emb = embed([c["content"] for c in chunks], "document")
        with self._lock:
            if self.has(name):
                logger.info("replacing existing document '%s'", name)
                self._remove(name)

            for s in range(0, len(chunks), BATCH):
                part = chunks[s : s + BATCH]
                self.client.upsert(
                    CHUNKS,
                    points=[
                        models.PointStruct(
                            id=_chunk_id(name, c["metadata"]["chunk_id"]),
                            vector=emb[s + j],
                            payload={
                                "name": name,
                                "chunk_id": c["metadata"]["chunk_id"],
                                "content": c["content"],
                            },
                        )
                        for j, c in enumerate(part)
                    ],
                )
            self.client.upsert(
                DOCS,
                points=[
                    models.PointStruct(
                        id=_doc_id(name),
                        vector=[1.0],
                        payload={"name": name, "text": text, **self._info(text)},
                    )
                ],
            )
        logger.info("stored %d vectors for '%s'", len(chunks), name)
        return len(chunks)

    def has(self, name: str) -> bool:
        return bool(self.client.retrieve(DOCS, ids=[_doc_id(name)], with_payload=False, with_vectors=False))

    def get_text(self, name: str) -> str:
        r = self.client.retrieve(DOCS, ids=[_doc_id(name)], with_payload=["text"], with_vectors=False)
        if not r:
            raise KeyError(name)
        return r[0].payload["text"]

    def info(self, name: str) -> dict[str, Any]:
        r = self.client.retrieve(DOCS, ids=[_doc_id(name)], with_payload=["chars", "sources"], with_vectors=False)
        if not r:
            raise KeyError(name)
        return dict(r[0].payload)

    def list_documents(self) -> list[str]:
        names: list[str] = []
        offset = None
        while True:
            pts, offset = self.client.scroll(
                DOCS, limit=256, offset=offset, with_payload=["name"], with_vectors=False
            )
            names.extend(p.payload["name"] for p in pts)
            if offset is None:
                break
        return sorted(names)

    def search(self, query_texts: list[str], k: int = 5) -> list[SearchHit]:
        query_texts = [q for q in query_texts if q.strip()]
        if not query_texts:
            raise ValueError("Query cannot be empty")

        vecs = embed(query_texts, "query")
        responses = self.client.query_batch_points(
            CHUNKS,
            requests=[models.QueryRequest(query=v, limit=k * 3, with_payload=True) for v in vecs],
        )
        total: dict[str, float] = {}
        best: dict[str, tuple[float, str]] = {}
        for resp in responses:
            per_query: dict[str, float] = {}
            for p in resp.points:
                n = p.payload["name"]
                s = float(p.score)
                if s > per_query.get(n, -1.0):
                    per_query[n] = s
                if s > best.get(n, (-1.0, ""))[0]:
                    best[n] = (s, p.payload["content"])
            for n, s in per_query.items():
                total[n] = total.get(n, 0.0) + s

        ranked = sorted(total, key=total.get, reverse=True)[:k]  # type: ignore[arg-type]
        return [
            {
                "name": n,
                "score": round(total[n] / len(query_texts), 4),
                "snippet": best[n][1][:300],
            }
            for n in ranked
        ]


@lru_cache(maxsize=1)
def get_store() -> QdrantVectorStore:
    return QdrantVectorStore()