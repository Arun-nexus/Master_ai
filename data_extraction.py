import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from configuration import get_params
from logger import logger
from model_call import ClaudeModel, get_model
from schemas import Chunk

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
SUPPORTED = (".pdf", ".docx", ".xlsx", ".pptx", *IMAGE_TYPES)

EXTRACT_PROMPT = (
    "Extract ALL the text, specs, numbers and tables from this document. "
    "Render tables as markdown. Return only the extracted content, no commentary."
)


PAGE_PROMPT = (
    "This is one page of a product catalogue / manual / tender document. Transcribe EVERYTHING on the page "
    "faithfully, reading both the text layer and what is visible (small print, tables, spec boxes, captions, "
    "text inside images and diagrams). Keep every value, range, unit, model name and option exactly as shown. "
    "Keep section headings. Render tables and multi-column spec blocks as markdown (one 'Parameter | Value' row "
    "per line, keeping each value with its own parameter). Mark optional items as (Optional). "
    "Do not summarise, do not guess unreadable text, skip repeated logos/addresses/footers. "
    "Return only the transcription."
)


class UnsupportedFormatError(ValueError):
    pass


class DocumentProcessor:
    def __init__(self, llm: Optional[ClaudeModel] = None) -> None:
        self.params = get_params()
        self._llm = llm

    @property
    def llm(self) -> ClaudeModel:
        if self._llm is None:
            self._llm = get_model()
        return self._llm

    # ---------- extraction ----------
    def extract(self, filename: str, content: bytes) -> str:
        ext = Path(filename).suffix.lower()
        try:
            if ext == ".pdf":
                text = self._pdf(content)
            elif ext in IMAGE_TYPES:
                text = self.llm.ask(EXTRACT_PROMPT, media=[(content, IMAGE_TYPES[ext])])
            elif ext == ".docx":
                text = self._docx(content)
            elif ext == ".xlsx":
                text = self._xlsx(content)
            elif ext == ".pptx":
                text = self._pptx(content)
            else:
                raise UnsupportedFormatError(
                    f"'{ext}' supported nahi hai. Allowed: {', '.join(SUPPORTED)}"
                )
        except UnsupportedFormatError:
            logger.error("unsupported file format: %s", ext)
            raise
        except Exception:
            logger.exception("extraction failed for %s", filename)
            raise

        text = text.strip()
        if not text:
            raise ValueError("Document se koi text extract nahi hua")
        logger.info("extracted %d chars from %s", len(text), filename)
        return text

    def _pdf(self, content: bytes) -> str:
        from pypdf import PdfReader, PdfWriter

        mode = os.getenv("PDF_MODE", "vision").lower()
        reader = PdfReader(io.BytesIO(content))
        pages: list[tuple[str, bytes | None]] = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if mode == "auto" and len(text) >= 800 and not self._has_images(page):
                pages.append((text, None))
                continue
            w = PdfWriter()
            w.add_page(page)
            buf = io.BytesIO()
            w.write(buf)
            pages.append((text, buf.getvalue()))

        def one(item: tuple[str, bytes | None]) -> str:
            text, data = item
            if data is None:
                return text
            try:
                return self.llm.ask(PAGE_PROMPT, media=[(data, "application/pdf")]).strip() or text
            except Exception:
                logger.exception("page extraction failed, using embedded text")
                return text

        with ThreadPoolExecutor(max_workers=int(os.getenv("PDF_WORKERS", "4"))) as ex:
            out = list(ex.map(one, pages))
        return "\n\n".join(f"### Page {i}\n{p}" for i, p in enumerate(out, 1) if p)

    @staticmethod
    def _has_images(page) -> bool:
        try:
            xo = page["/Resources"].get("/XObject")
            if not xo:
                return False
            return any(xo[k].get_object().get("/Subtype") == "/Image" for k in xo)
        except Exception:
            return True

    @staticmethod
    def _docx(content: bytes) -> str:
        from docx import Document

        doc = Document(io.BytesIO(content))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        return "\n".join(parts)

    @staticmethod
    def _xlsx(content: bytes) -> str:
        from openpyxl import load_workbook

        wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        parts: list[str] = []
        for ws in wb.worksheets:
            parts.append(f"## {ws.title}")
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)

    @staticmethod
    def _pptx(content: bytes) -> str:
        from pptx import Presentation

        prs = Presentation(io.BytesIO(content))
        parts: list[str] = []
        for i, slide in enumerate(prs.slides, 1):
            parts.append(f"## Slide {i}")
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text.strip():
                    parts.append(shape.text_frame.text)
        return "\n".join(parts)

    # ---------- chunking ----------
    def chunk(self, text: str, name: str) -> list[Chunk]:
        size = self.params["chunk_size"]
        step = max(1, size - self.params["chunk_overlap"])
        chunks: list[Chunk] = []
        for i in range(0, len(text), step):
            piece = text[i : i + size]
            if piece.strip():
                chunks.append(
                    {"content": piece, "metadata": {"name": name, "chunk_id": len(chunks)}}
                )
        return chunks