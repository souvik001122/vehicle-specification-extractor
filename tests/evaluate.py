"""Check the pipeline against hand-verified answers (same logic as the app's "Batch check" tab).

    python tests/evaluate.py                         # manual in data/
    python tests/evaluate.py --source my.pdf         # an uploaded / real PDF
    python tests/evaluate.py --embed-backend sbert   # the embeddings used in the app

``tests/expected_queries.csv`` holds one expected answer per row (the value was read from the manual
text; ``manual_line`` is its line number in data/sample-service-manual.txt).  See ``src/evaluation.py``
for the pass rules.  Prints PASS/FAIL per row and a summary; exit code 1 if anything fails.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import FAIL, load_expectations, run_batch, summarize  # noqa: E402
from src.main import Pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--embed-backend", default=None)
    ap.add_argument("--csv", default=str(ROOT / "tests" / "expected_queries.csv"))
    args = ap.parse_args()

    rows = load_expectations(Path(args.csv).read_bytes())
    pipe = Pipeline(args.source, embed_backend=args.embed_backend, extract_backend="rules")
    print(f"source={pipe.source.name} chunks={len(pipe.chunks)} embeddings={pipe.retriever.embedder.backend} "
          f"text-extraction={pipe.extraction.get('strategy')} {pipe.extraction.get('scores') or ''}\n")
    report = run_batch(pipe, rows)
    for i, r in enumerate(report, 1):
        print(f"{i:2d} {r['status']:4s}  {r['query']}\n       expected: {r['expected']}")
        if r["status"] == FAIL:
            print(f"       got {r['results']} result(s): {r['got'] or '-'}")
    s = summarize(report)
    print(f"\nRESULT: {s['passed']}/{s['checks']} expectations passed "
          f"({s['queries_fully_correct']}/{s['queries']} queries fully correct)")
    return 0 if s["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
