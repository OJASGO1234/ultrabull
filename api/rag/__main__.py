import argparse
import sys

from . import config
from .embedder import FastEmbedder
from .ingest import ingest
from .retrieve import search
from .store import Store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m api.rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("--rebuild", action="store_true", help="drop the index and re-embed everything")
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    p.add_argument("--min-score", type=float, default=config.DEFAULT_MIN_SCORE)
    sub.add_parser("status")
    a = ap.parse_args(argv)

    store = Store()
    try:
        if a.cmd == "ingest":
            r = ingest(store, FastEmbedder(), rebuild=a.rebuild)
            print(f"added {len(r.added)}, updated {len(r.updated)}, unchanged {len(r.unchanged)}, "
                  f"removed {len(r.removed)}, failed {len(r.failed)} -> {r.total_chunks} chunks")
            for f, err in r.failed.items():
                print(f"  FAILED {f}: {err}")
            return 1 if r.failed else 0
        if a.cmd == "status":
            print(f"model={store.get_meta('model')} chunks={store.count()}")
            for s, n in store.sources():
                print(f"  {n:4d}  {s}")
            return 0
        hits = search(store, FastEmbedder(), a.query, a.k, a.min_score)
        if not hits:
            print("no results above the score threshold")
        for h in hits:
            loc = h.heading or (f"p.{h.page}" if h.page else "")
            print(f"[{h.score:.3f}] {h.source} :: {loc}\n    {h.text[:200]!r}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
