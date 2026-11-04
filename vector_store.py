import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from configuration import BASE_DIR, get_params
from logger import logger
from schemas import Chunk, SearchHit

SRC_RE = re.compile(r"(?m)^## \[(catalogue|manual)\] (.+)$")


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Embedding model sirf ek baar load hota hai (pehle har request pe load hota tha)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = get_params()["embedding_model"]
    logger.info("loading embedding model %s on %s", name, device)
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = 512  # chunk 1000 chars (~300 tokens) hai; default 8192 activations bhaari karta hai
    if device == "cuda":
        model.half()  # fp16: weights ~2.2GB -> ~1.1GB
    return model


class FAISSVectorStore:
    """Disk par persist hota hai. RAM me sirf vectors + document ke naam/chhoti info rehti hai.
    Poora text alag file me disk par rehta hai aur get_text() call hone par hi padha jata hai."""

    def __init__(self) -> None:
        p = get_params()
        self.model = get_embedder()
        self._lock = RLock()
        self.size = p["chunk_size"]
        self.step = max(1, p["chunk_size"] - p["chunk_overlap"])

        storage = Path(p["storage_path"])
        if not storage.is_absolute():
            storage = BASE_DIR / storage
        storage.mkdir(parents=True, exist_ok=True)
        self.index_path = storage / "index.faiss"
        self.docs_path = storage / "documents.json"
        self.text_dir = storage / "texts"
        self.text_dir.mkdir(exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.meta: list[tuple[str, int]] = []  # har vector ke liye (doc name, chunk_id) - text nahi
        self.docs: dict[str, dict[str, Any]] = {}  # name -> {"chars", "sources"}
        self._load()

    # ---------- persistence ----------
    def _text_path(self, name: str) -> Path:
        return self.text_dir / f"{hashlib.sha1(name.encode('utf-8')).hexdigest()}.txt"

    @staticmethod
    def _info(text: str) -> dict[str, Any]:
        return {"chars": len(text), "sources": [{"kind": k, "file": f} for k, f in SRC_RE.findall(text)]}

    def _load(self) -> None:
        if not (self.index_path.exists() and self.docs_path.exists()):
            return
        self.index = faiss.read_index(str(self.index_path))
        data = json.loads(self.docs_path.read_text(encoding="utf-8"))
        if "texts" in data:  # purana format: saara text ek JSON me -> alag files me migrate
            self.meta = [(c["metadata"]["name"], c["metadata"]["chunk_id"]) for c in data["chunks"]]
            for n, t in data["texts"].items():
                self._write_text(n, t)
                self.docs[n] = self._info(t)
            self._persist()
            logger.info("migrated old store: %d docs moved to per-file text", len(self.docs))
        else:
            self.meta = [(n, int(i)) for n, i in data["meta"]]
            self.docs = data["docs"]
        logger.info("loaded index: %d vectors, %d docs", self.index.ntotal, len(self.docs))

    def _write_text(self, name: str, text: str) -> None:
        path = self._text_path(name)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _persist(self) -> None:
        assert self.index is not None
        faiss.write_index(self.index, str(self.index_path))
        tmp = self.docs_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"meta": self.meta, "docs": self.docs}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.docs_path)

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
                torch.cuda.empty_cache()  # allocator ka cached memory wapas free
        return np.asarray(emb, dtype=np.float32)

    # ---------- write ----------
    def _remove(self, name: str) -> None:
        keep = [i for i, (n, _) in enumerate(self.meta) if n != name]
        if self.index is not None and self.index.ntotal:
            vecs = self.index.reconstruct_n(0, self.index.ntotal)
            new_index = faiss.IndexFlatIP(self.index.d)
            if keep:
                new_index.add(vecs[keep])
            self.index = new_index
        self.meta = [self.meta[i] for i in keep]
        self.docs.pop(name, None)
        self._text_path(name).unlink(missing_ok=True)

    def add(self, name: str, text: str, chunks: list[Chunk]) -> int:
        if not chunks:
            raise ValueError("Chunks list empty hai")
        with self._lock:
            if name in self.docs:  # same naam = replace
                logger.info("replacing existing document '%s'", name)
                self._remove(name)

            emb = self._encode([c["content"] for c in chunks])
            if self.index is None:
                self.index = faiss.IndexFlatIP(emb.shape[1])
            elif self.index.d != emb.shape[1]:
                raise ValueError("Embedding dimension mismatch")

            self.index.add(emb)
            self.meta.extend((c["metadata"]["name"], c["metadata"]["chunk_id"]) for c in chunks)
            self._write_text(name, text)
            self.docs[name] = self._info(text)
            self._persist()
        logger.info("stored %d vectors for '%s'", len(chunks), name)
        return len(chunks)

    # ---------- read ----------
    def has(self, name: str) -> bool:
        return name in self.docs

    def get_text(self, name: str) -> str:
        """Text disk se tabhi padha jata hai jab call ho; caller ke baad memory free ho jati hai."""
        if name not in self.docs:
            raise KeyError(name)
        return self._text_path(name).read_text(encoding="utf-8")

    def info(self, name: str) -> dict[str, Any]:
        return self.docs[name]

    def list_documents(self) -> list[str]:
        return sorted(self.docs)

    def _snippet(self, name: str, chunk_id: int) -> str:
        try:
            start = chunk_id * self.step
            return self.get_text(name)[start : start + self.size][:300]
        except (KeyError, OSError):
            return ""

    def search(self, query_texts: list[str], k: int = 5) -> list[SearchHit]:
        """Query ke chunks ko search karke document-level ranking return karta hai."""
        query_texts = [q for q in query_texts if q.strip()]
        if not query_texts:
            raise ValueError("Query cannot be empty")

        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            scores, idxs = self.index.search(
                self._encode(query_texts), min(k * 3, self.index.ntotal)
            )
            total: dict[str, float] = {}
            best: dict[str, tuple[float, int]] = {}
            for row_s, row_i in zip(scores, idxs):
                per_query: dict[str, float] = {}
                for s, i in zip(row_s, row_i):
                    if i < 0:
                        continue
                    n, cid = self.meta[int(i)]
                    s = float(s)
                    if s > per_query.get(n, -1.0):
                        per_query[n] = s
                    if s > best.get(n, (-1.0, 0))[0]:
                        best[n] = (s, cid)
                for n, s in per_query.items():
                    total[n] = total.get(n, 0.0) + s

        ranked = sorted(total, key=total.get, reverse=True)[:k]  # type: ignore[arg-type]
        return [
            {
                "name": n,
                "score": round(total[n] / len(query_texts), 4),
                "snippet": self._snippet(n, best[n][1]),
            }
            for n in ranked
        ]


@lru_cache(maxsize=1)
def get_store() -> FAISSVectorStore:
    return FAISSVectorStore()