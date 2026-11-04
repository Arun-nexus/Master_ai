import base64
from functools import lru_cache
from typing import Iterator, Optional, Sequence, Tuple

import anthropic

from configuration import get_params
from logger import logger

Media = Tuple[bytes, str]  # (raw bytes, mime type)


class ClaudeModel:
    def __init__(self) -> None:
        p = get_params()
        self.client = anthropic.Anthropic(max_retries=6, timeout=300.0)
        self.model = p["model"]
        self.max_tokens = p["max_tokens"]

    @staticmethod
    def _media_block(data: bytes, mime: str) -> dict:
        kind = "document" if mime == "application/pdf" else "image"
        return {
            "type": kind,
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": base64.standard_b64encode(data).decode(),
            },
        }

    def _messages(self, prompt: str, media: Optional[Sequence[Media]]) -> list[dict]:
        content: list[dict] = [self._media_block(d, m) for d, m in (media or [])]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        media: Optional[Sequence[Media]] = None,
    ) -> Iterator[str]:
        """Token-by-token streaming generator."""
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._messages(prompt, media),
        }
        if system:
            kwargs["system"] = system
        try:
            with self.client.messages.stream(**kwargs) as s:
                for text in s.text_stream:
                    yield text
        except anthropic.APIError:
            logger.exception("Claude API call failed")
            raise

    def ask(
        self,
        prompt: str,
        system: Optional[str] = None,
        media: Optional[Sequence[Media]] = None,
    ) -> str:
        """Non-streaming helper (extraction ke liye)."""
        return "".join(self.stream(prompt, system=system, media=media))


@lru_cache(maxsize=1)
def get_model() -> ClaudeModel:
    return ClaudeModel()