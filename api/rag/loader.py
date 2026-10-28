"""Find allowed documents and split each into sections (text + heading path + page)."""
from __future__ import annotations
import glob
import os
import re
from dataclasses import dataclass

from . import config


@dataclass
class Section:
    text: str
    heading: str = ""        # "Title > Sub heading" path, "" if none
    page: int | None = None  # 1-based, PDFs only


def discover(root: str = config.ROOT, globs: list[str] | None = None) -> list[str]:
    """Root-relative posix paths of every allowed document, sorted."""
    found: set[str] = set()
    for pattern in globs or config.SOURCE_GLOBS:
        for path in glob.glob(os.path.join(root, pattern), recursive=True):
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            parts = rel.split("/")
            name = parts[-1]
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in config.ALLOWED_EXT:
                continue
            if name in config.EXCLUDED_FILES or config.SECRET_NAME.search(name):
                continue
            if any(p in config.EXCLUDED_DIRS for p in parts[:-1]):
                continue
            found.add(rel)
    return sorted(found)


_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _strip_frontmatter(text: str) -> tuple[str, str]:
    """(body, title) -- MDX/Markdown YAML frontmatter, plus MDX import/export lines."""
    title = ""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            m = re.search(r"^title:\s*(.+)$", text[3:end], re.M)
            if m:
                title = m.group(1).strip().strip("'\"")
            text = text[end + 4:]
    text = re.sub(r"^(import|export)\s.*$", "", text, flags=re.M)
    return text, title


def sections_from_markdown(text: str, default_title: str = "") -> list[Section]:
    text, fm_title = _strip_frontmatter(text.replace("\r\n", "\n"))
    stack: list[tuple[int, str]] = []
    if fm_title or default_title:
        stack.append((0, fm_title or default_title))
    out: list[Section] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            out.append(Section(body, " > ".join(t for _, t in stack)))
        buf.clear()

    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        else:
            buf.append(line)
    flush()
    return out


def sections_from_pdf(path: str) -> list[Section]:
    from pypdf import PdfReader  # lazy: only needed if a PDF is present
    out = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append(Section(text, page=i))
    return out


def load(rel_path: str, root: str = config.ROOT) -> list[Section]:
    path = os.path.join(root, rel_path)
    ext = os.path.splitext(rel_path)[1].lower()
    if ext == ".pdf":
        return sections_from_pdf(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if ext == ".txt":
        return [Section(raw.strip())] if raw.strip() else []
    return sections_from_markdown(raw, default_title=os.path.splitext(os.path.basename(rel_path))[0])
