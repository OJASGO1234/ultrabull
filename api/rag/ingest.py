"""Idempotent ingestion: unchanged files are skipped, changed files are replaced
wholesale, deleted files are pruned. Chunk ids are content hashes, so re-running
can never create duplicates."""
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass, field

from . import config
from .chunker import chunk_document
from .embedder import Embedder
from .loader import discover, load
from .store import Store


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    total_chunks: int = 0


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest(store: Store, embedder: Embedder, root: str = config.ROOT,
           globs: list[str] | None = None, rebuild: bool = False) -> IngestReport:
    report = IngestReport()
    if rebuild:
        store.clear()
    stored_model = store.get_meta("model")
    if stored_model and stored_model != embedder.name:
        raise RuntimeError(
            f"index was built with {stored_model!r} but the embedder is {embedder.name!r}; "
            "re-run with rebuild=True (CLI: --rebuild) so vectors are never mixed across models"
        )

    known = store.doc_hashes()
    current = discover(root, globs)
    for gone in sorted(set(known) - set(current)):
        store.delete_document(gone)
        report.removed.append(gone)

    for rel in current:
        try:
            sha = _sha(os.path.join(root, rel))
            if known.get(rel) == sha:
                report.unchanged.append(rel)
                continue
            chunks = chunk_document(rel, load(rel, root))
            vectors = embedder.embed([c.embed_text() for c in chunks]) if chunks else []
            store.replace_document(rel, sha, chunks, vectors)
            (report.updated if rel in known else report.added).append(rel)
        except Exception as e:  # one bad file must not abort the run
            report.failed[rel] = f"{type(e).__name__}: {e}"
    store.set_meta("model", embedder.name)
    report.total_chunks = store.count()
    return report
