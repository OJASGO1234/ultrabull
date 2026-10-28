"""Paragraph-packing chunker. Never splits mid-word unless a single token is huge."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass

from . import config
from .loader import Section


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    page: int | None
    index: int
    text: str

    def embed_text(self) -> str:
        """What gets embedded: the heading path gives the passage its context."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def _split_long(block: str, limit: int) -> list[str]:
    """Break an oversize block on line boundaries, then on spaces."""
    pieces, cur = [], ""
    for line in block.split("\n"):
        while len(line) > limit:
            cut = line.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(line[:cut])
            line = line[cut:].lstrip()
        if cur and len(cur) + len(line) + 1 > limit:
            pieces.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        pieces.append(cur)
    return pieces


def chunk_section(sec: Section, limit: int = config.CHUNK_MAX_CHARS) -> list[str]:
    paras = [p.strip() for p in sec.text.split("\n\n") if p.strip()]
    out, cur = [], ""
    for p in paras:
        for piece in (_split_long(p, limit) if len(p) > limit else [p]):
            if cur and len(cur) + len(piece) + 2 > limit:
                out.append(cur)
                cur = piece
            else:
                cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        out.append(cur)
    return out


def chunk_document(source: str, sections: list[Section]) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for sec in sections:
        for text in chunk_section(sec):
            if len(text) < config.CHUNK_MIN_CHARS:
                continue  # stray headings / page numbers add noise, not signal
            if config.SECRET_TEXT.search(text):
                continue  # looks like a credential: never embed or store it
            key = "\x1f".join([source, sec.heading, str(sec.page), text])
            cid = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
            if cid in seen:  # identical text under the same heading: keep one
                continue
            seen.add(cid)
            chunks.append(Chunk(cid, source, sec.heading, sec.page, len(chunks), text))
    return chunks
