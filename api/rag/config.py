"""RAG settings. Everything generated lives under data_cache/rag/ (git-ignored)."""
from __future__ import annotations
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAG_DIR = os.environ.get("RAG_DIR", os.path.join(ROOT, "data_cache", "rag"))
DB_PATH = os.path.join(RAG_DIR, "rag.db")
MODEL_CACHE = os.path.join(RAG_DIR, "models")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Allowlist, not a crawl: only these (root-relative) globs are ever read.
SOURCE_GLOBS = [
    "core/*.md",
    "docs/content/**/*.md",
    "docs/content/**/*.mdx",
    "*.pdf",
    "core/*.pdf",
    "reports/**/*.md",
    "reports/**/*.pdf",
]
# Extensions the loader understands (a glob can never widen this).
ALLOWED_EXT = {".md", ".mdx", ".txt", ".pdf"}
# Files that match a glob but are boilerplate, not project knowledge.
EXCLUDED_FILES = {"chainlit.md"}
EXCLUDED_DIRS = {"venv", ".venv", "node_modules", "__pycache__", ".git", "data_cache", ".files"}
SECRET_NAME = re.compile(r"(^\.env)|secret|token|credential|password|\.pem$|\.key$", re.I)
# Chunks that look like they contain a credential are dropped, never embedded.
SECRET_TEXT = re.compile(
    r"sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{32,}|eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}"
    r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)

CHUNK_MAX_CHARS = 1200
CHUNK_MIN_CHARS = 80
DEFAULT_TOP_K = 5
# Calibrated against bge-small on this corpus: unrelated queries top out ~0.45-0.5.
DEFAULT_MIN_SCORE = 0.55
