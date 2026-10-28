"""RAG answer-generation tests (stub Claude client, real index)."""
from types import SimpleNamespace

from .rag.embedder import FastEmbedder
from .rag.generate import ask
from .rag.store import Store


class _Client:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Stubbed answer [1].")])


def test_ask_cites_retrieved_passages():
    store, client = Store(), _Client()
    try:
        out = ask(store, FastEmbedder(), client, "was there any look-ahead bias in the audit?")
    finally:
        store.close()
    assert out["answer"] == "Stubbed answer [1]."
    assert out["sources"] and out["sources"][0]["n"] == 1
    prompt = client.calls[0]["messages"][0]["content"]
    assert "[1] " in prompt and out["sources"][0]["text"] in prompt


def test_ask_skips_claude_when_nothing_relevant():
    store, client = Store(), _Client()
    try:
        out = ask(store, FastEmbedder(), client, "best recipe for chocolate cake")
    finally:
        store.close()
    assert out["sources"] == [] and client.calls == []
