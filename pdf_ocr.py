import io
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
import fitz
import numpy as np
import torch
from PIL import Image, ImageOps
from fastapi import UploadFile
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

MODELS_DIR = Path(__file__).parent / "models"
EMBED_MODEL = "BAAI/bge-m3"
OCR_DPI = 300
OCR_RETRY_DPI = 450
MIN_NATIVE_CHARS = 30
MIN_IMAGE_AREA = 0.05
MIN_CHUNK_CHARS = 400
MAX_CHUNK_CHARS = 3000
MIN_IMAGE_SIDE = 1600
OCR_WORKERS = int(os.getenv("OCR_WORKERS", str(min(4, os.cpu_count() or 1))))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))

logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logging.getLogger("RapidOCR").setLevel(logging.ERROR)
log = logging.getLogger("pdf_ocr")


def _make_embeddings() -> HuggingFaceEmbeddings:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available() and os.getenv("EMBED_DEVICE", "cuda") == "cuda"
    model_kwargs = {"device": "cuda" if use_cuda else "cpu"}
    if use_cuda:
        model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        cache_folder=str(MODELS_DIR),
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True, "batch_size": EMBED_BATCH},
    )


embeddings = _make_embeddings()
_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

_tls = threading.local()
_ocr_pool = ThreadPoolExecutor(OCR_WORKERS, thread_name_prefix="ocr")


def _get_ocr():
    if not hasattr(_tls, "ocr"):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            from rapidocr import RapidOCR
        _tls.ocr = RapidOCR()
    return _tls.ocr


def _ocr_image(img: np.ndarray) -> str:
    out = _get_ocr()(img)
    if hasattr(out, "txts"):
        return "\n".join(out.txts) if out.txts else ""
    result = out[0] if isinstance(out, tuple) else out
    return "\n".join(r[1] for r in result) if result else ""


def _render(page: "fitz.Page", dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        img = np.repeat(img, 3, axis=2)
    return np.pad(img, ((20, 20), (20, 20), (0, 0)), constant_values=255)


def _image_area_ratio(page: "fitz.Page") -> float:
    page_area = page.rect.width * page.rect.height or 1
    total = 0.0
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        total += max(0, x1 - x0) * max(0, y1 - y0)
    return total / page_area


def _merge(native: str, ocr: str) -> str:
    """OCR ki sirf wahi lines jodo jo native text mein pehle se nahi hain."""
    seen = {" ".join(l.split()).lower() for l in native.splitlines() if l.strip()}
    extra = [l for l in ocr.splitlines() if l.strip() and " ".join(l.split()).lower() not in seen]
    return (native + "\n" + "\n".join(extra)).strip() if extra else native


def _pdf_pages(data: bytes, name: str) -> List[str]:
    with fitz.open(stream=data, filetype="pdf") as pdf:
        n = len(pdf)
        results = [""] * n
        pending = [] 

        def flush():
            if not pending:
                return
            texts = list(_ocr_pool.map(_ocr_image, [p[2] for p in pending]))
            for (i, native, _), t in zip(pending, texts):
                if not t:
                    t = _ocr_image(_render(pdf[i], OCR_RETRY_DPI))
                results[i] = _merge(native, t) if native and t else (t or native)
            pending.clear()

        for i in range(n):
            page = pdf[i]
            native = page.get_text("text").strip()
            if len(native) >= MIN_NATIVE_CHARS and _image_area_ratio(page) < MIN_IMAGE_AREA:
                results[i] = native
                continue
            pending.append((i, native, _render(page, OCR_DPI)))
            if len(pending) >= OCR_WORKERS * 2:  
                flush()
        flush()
        log.info("[%s] %d pages, %d non-empty", name, n, sum(1 for r in results if r))
        return [r for r in results if r]


def _open_image(data: bytes):
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        return None


def _image_text(img: Image.Image, name: str) -> str:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) < MIN_IMAGE_SIDE:
        s = MIN_IMAGE_SIDE / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    arr = np.pad(np.array(img), ((20, 20), (20, 20), (0, 0)), constant_values=255)
    text = _ocr_image(arr)
    log.info("[%s] image OCR: %d chars", name, len(text))
    return text


def _read(upload: UploadFile) -> bytes:
    upload.file.seek(0)
    return upload.file.read()


def extract_text(upload: UploadFile) -> str:
    data = _read(upload)
    name = upload.filename or "file"
    if data[:5] == b"%PDF-":
        return "\n\n".join(_pdf_pages(data, name))
    img = _open_image(data)
    if img is not None:
        return _image_text(img, name)
    return data.decode("utf-8", errors="ignore")


def _split_long(chunk: str) -> List[str]:
    if len(chunk) <= MAX_CHUNK_CHARS:
        return [chunk]
    out, cur = [], ""
    for line in chunk.splitlines():
        while len(line) > MAX_CHUNK_CHARS:
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:MAX_CHUNK_CHARS])
            line = line[MAX_CHUNK_CHARS:]
        if cur and len(cur) + len(line) + 1 > MAX_CHUNK_CHARS:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return out


def _merge_small(chunks: List[str]) -> List[str]:
    out, cur = [], ""
    for c in chunks:
        if cur and len(cur) + len(c) + 2 > MAX_CHUNK_CHARS:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{c}" if cur else c
        if len(cur) >= MIN_CHUNK_CHARS:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def extract_chunks(upload: UploadFile) -> List[str]:
    text = extract_text(upload).strip()
    if not text:
        return []
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts = [p.strip() for p in _chunker.split_text(text) if p.strip()]
    sized = [piece for p in parts for piece in _split_long(p)]
    chunks = [c.strip() for c in _merge_small(sized) if c.strip()]
    log.info("[%s] %d chars -> %d chunks", upload.filename, len(text), len(chunks))
    return chunks