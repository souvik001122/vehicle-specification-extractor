"""Check the extractor against expected_queries.json and print a PASS/FAIL report.

    python tests/check_expected.py                          # sample manual in data/
    python tests/check_expected.py --source my_manual.pdf   # a PDF (or .txt) of your own
    python tests/check_expected.py --embed sbert            # use the sentence-transformers embeddings

Paste the whole printed report when asking for a fix: it names the strategy used to read the PDF and,
for every failing query, what was returned instead.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extractor import public  # noqa: E402
from src.main import Pipeline  # noqa: E402


def matches(row: dict, want: dict) -> bool:
    return (want.get("component", "").lower() in row["component"].lower()
            and (not want.get("value") or row["value"] == want["value"])
            and (not want.get("unit") or row["unit"] == want["unit"])
            and (not want.get("section") or want["section"] in row["section"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--embed", default=None, help="tfidf | sbert | openai (default: auto)")
    ap.add_argument("--file", default=str(ROOT / "expected_queries.json"))
    args = ap.parse_args()
    pipe = Pipeline(args.source, extract_backend="rules", embed_backend=args.embed)
    print(f"source={pipe.source.name} pages={len(pipe.pages)} chunks={len(pipe.chunks)} "
          f"embeddings={pipe.retriever.embedder.backend} extraction={pipe.extraction.get('strategy')} "
          f"scores={pipe.extraction.get('scores')}")
    failed = 0
    for case in json.loads(Path(args.file).read_text(encoding="utf-8")):
        recs, _, used = pipe.run(case["query"])
        rows = [public(r) for r in recs]
        problems = []
        if case.get("expect_empty") and rows:
            problems.append("expected no result")
        for w in case.get("must_include", []):
            if not any(matches(r, w) for r in rows):
                problems.append(f"missing {w}")
        for w in case.get("must_not_include", []):
            if any(matches(r, w) for r in rows):
                problems.append(f"should not contain {w}")
        print(("PASS " if not problems else "FAIL ") + f"[{len(rows)} rows, {used}] {case['query']}")
        if problems:
            failed += 1
            for p in problems:
                print("     -", p)
            for r in rows[:8]:
                print("     got:", r["component"], "|", r["value"], r["unit"], "|", r["section"])
    print(f"\n{len(json.loads(Path(args.file).read_text(encoding='utf-8'))) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
