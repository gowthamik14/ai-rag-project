"""
Simple recursive character-based text splitter.

Splits on paragraph boundaries first, then sentences, then words,
to preserve semantic coherence within each chunk.
"""
from __future__ import annotations

from typing import List, Optional

from config.settings import settings


class TextSplitter:
    """
    Splits text into overlapping chunks suitable for embedding.

    chunk_size    – target maximum characters per chunk
    chunk_overlap – characters shared between adjacent chunks
    """

    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def split(self, text: str) -> List[str]:
        """Return a list of text chunks."""
        text = text.strip()
        if not text:
            return []
        chunks = self._split_recursive(text, self._SEPARATORS)
        return [c.strip() for c in chunks if c.strip()]

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # Last resort: hard character split
            return self._hard_split(text)

        parts = text.split(sep)

        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).lstrip(sep) if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                # current chunk is full – flush it
                if current:
                    if len(current) > self.chunk_size and remaining_seps:
                        chunks.extend(self._split_recursive(current, remaining_seps))
                    else:
                        chunks.append(current)
                current = part

                # Handle a single part that is already too long
                if len(current) > self.chunk_size and remaining_seps:
                    chunks.extend(self._split_recursive(current, remaining_seps))
                    current = ""

        if current:
            if len(current) > self.chunk_size and remaining_seps:
                chunks.extend(self._split_recursive(current, remaining_seps))
            else:
                chunks.append(current)

        return self._apply_overlap(chunks)

    def _hard_split(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = overlapped[-1][-self.chunk_overlap :]
            overlapped.append(prev_tail + " " + chunks[i])

        return overlapped
