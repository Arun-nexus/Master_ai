import os
import re
import uuid
from functools import lru_cache
from threading import RLock
from typing import Any

import numpy as np
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from configuration import get_params
from logger import logger
from schemas import Chunk, SearchHit

SRC_RE = re.compile(r"(?m)^## \[(catalogue|manual)\] (.+)$")
CHUNKS = "chunks"
DOCS = "docs"
BATCH = 64


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = get_params()["embedding_model"]
    logger.info("loading embedding model %s on %s", name, device)
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = 512
    if device == "cuda":
        model.half()
    return model


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

    @property
    def model(self) -> SentenceTransformer:
        return get_embedder()

    def _init_collections(self) -> None:
        dim = int(os.getenv("EMBED_DIM", "1024"))
        if not self.client.collection_exists(CHUNKS):
            self.client.create_collection(
                CHUNKS,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
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

    def _encode(self, texts: list[str]) -> np.ndarray:
        try:
            with torch.inference_mode():
                emb = self.model.encode(
                    texts,
                    batch_size=8,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.asarray(emb, dtype=np.float32)

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
        with self._lock:
            if self.has(name):
                logger.info("replacing existing document '%s'", name)
                self._remove(name)

            emb = self._encode([c["content"] for c in chunks])
            for s in range(0, len(chunks), BATCH):
                part = chunks[s : s + BATCH]
                self.client.upsert(
                    CHUNKS,
                    points=[
                        models.PointStruct(
                            id=_chunk_id(name, c["metadata"]["chunk_id"]),
                            vector=emb[s + j].tolist(),
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

        vecs = self._encode(query_texts)
        responses = self.client.query_batch_points(
            CHUNKS,
            requests=[
                models.QueryRequest(query=v.tolist(), limit=k * 3, with_payload=True)
                for v in vecs
            ],
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