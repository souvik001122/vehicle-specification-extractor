"""Pipeline + command line interface.

    python -m src.main --query "Torque for brake caliper bolts"
    python -m src.main --queries-file queries.txt --out outputs/results.csv      # or .json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):                       # allow `python src/main.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

import config  # noqa: E402
from .chunking import Chunk, chunk_documents  # noqa: E402
from .embeddings import Embedder  # noqa: E402
from .extractor import Extractor, Record, public  # noqa: E402
from .pdf_extract import Doc, load_documents  # noqa: E402
from .retriever import Retriever  # noqa: E402


def default_source() -> Path:
    """The manual in data/ (a text dump is preferred when both a .txt and a .pdf exist)."""
    for pattern in ("*.txt", "*.pdf"):
        found = sorted(config.DATA_DIR.glob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError(f"No manual found in {config.DATA_DIR}. Pass --source path/to/manual.pdf")


class Pipeline:
    """extract text -> chunk -> embed/index -> retrieve -> extract structured specs."""

    def __init__(self, source: str | Path | None = None, extract_backend: str | None = None,
                 embed_backend: str | None = None):
        self.source = Path(source) if source else default_source()
        self.docs: list[Doc] = load_documents(self.source)
        self.pages: list[str] = [p for d in self.docs for p in d.pages]
        self.chunks: list[Chunk] = chunk_documents(self.docs)
        self.retriever = Retriever(self.chunks, Embedder(embed_backend))
        self.extractor = Extractor(extract_backend)
        self.extractor.index_corpus(self.chunks)

    def run(self, query: str, top_k: int | None = None) -> tuple[list[Record], list[tuple[Chunk, float]], str]:
        """Return (records, retrieved (chunk, score) pairs, extractor used)."""
        hits = self.retriever.search(query, top_k or config.TOP_K)
        records, used = self.extractor.extract(query, hits)
        return records, hits, used


# --------------------------------------------------------------------------- output
def write_output(path: Path, results: list[tuple[str, list[dict]]], single: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        fields = ["query", "component", "spec_type", "value", "unit", "section"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for q, rows in results:
                for r in rows:
                    w.writerow({"query": q, **r})
    else:
        payload = results[0][1] if single else [{"query": q, "results": rows} for q, rows in results]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_queries(args) -> list[str]:
    queries = list(args.query or [])
    if args.queries_file:
        queries += [ln.strip() for ln in Path(args.queries_file).read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#")]
    return queries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract vehicle specifications from a service manual (RAG).")
    ap.add_argument("--query", "-q", action="append", help="question, e.g. 'Torque for brake caliper bolts' (repeatable)")
    ap.add_argument("--queries-file", help="text file with one question per line")
    ap.add_argument("--source", help="manual (.pdf or .txt); default: first file in data/")
    ap.add_argument("--out", help="write results to this .json or .csv file")
    ap.add_argument("--top-k", type=int, default=config.TOP_K, help="chunks to retrieve (default %(default)s)")
    ap.add_argument("--extract-backend", choices=["auto", "rules", "openai", "anthropic"], default=None)
    ap.add_argument("--embed-backend", choices=["auto", "tfidf", "sbert", "openai"], default=None)
    ap.add_argument("--minimal", action="store_true", help="only component/spec_type/value/unit (no section)")
    ap.add_argument("--show-chunks", action="store_true", help="print the retrieved chunks to stderr")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    queries = _read_queries(args)
    if not queries:
        ap.error("give --query or --queries-file")

    pipe = Pipeline(args.source, extract_backend=args.extract_backend, embed_backend=args.embed_backend)
    print(f"[{pipe.source.name}] {len(pipe.pages)} pages, {len(pipe.docs)} sections-docs, {len(pipe.chunks)} chunks | "
          f"embeddings={pipe.retriever.embedder.backend}/{pipe.retriever.index.kind} | extractor={pipe.extractor.name}",
          file=sys.stderr)

    results: list[tuple[str, list[dict]]] = []
    for q in queries:
        records, hits, used = pipe.run(q, args.top_k)
        rows = [public(r, args.minimal) for r in records]
        if args.minimal:                                    # without the section, front/rear rows can be identical
            rows = [dict(t) for t in dict.fromkeys(tuple(r.items()) for r in rows)]
        results.append((q, rows))
        print(f"[{q}] {len(rows)} result(s) via {used}", file=sys.stderr)
        if args.show_chunks:
            for c, s in hits:
                print(f"   {s:.4f} {c.kind:14s} {c.section} | {c.heading}", file=sys.stderr)

    if args.out:
        write_output(Path(args.out), results, single=len(results) == 1)
        print(f"written to {args.out}", file=sys.stderr)
    else:
        payload = results[0][1] if len(results) == 1 else [{"query": q, "results": r} for q, r in results]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
