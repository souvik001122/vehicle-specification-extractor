"""Check the pipeline against hand-verified answers.

    python tests/evaluate.py                         # manual in data/
    python tests/evaluate.py --source my.pdf         # an uploaded / real PDF
    python tests/evaluate.py --embed-backend sbert   # the embeddings used in the app

``tests/expected_queries.csv`` holds one expected answer per row (the value was read from the manual
text; ``manual_line`` is its line number in data/sample-service-manual.txt).  A row passes when the
pipeline returns a result whose component contains ``component_contains``, with the same value, unit
and a section that contains ``section_contains``.  ``NONE`` means the manual has no answer, so the
pipeline must return nothing.  Prints PASS/FAIL per row and a summary; exit code 1 if anything fails.
"""
import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extractor import public  # noqa: E402
from src.main import Pipeline  # noqa: E402


def norm(v: str) -> str:
    return v.replace(",", "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--embed-backend", default=None)
    ap.add_argument("--csv", default=str(ROOT / "tests" / "expected_queries.csv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    pipe = Pipeline(args.source, embed_backend=args.embed_backend, extract_backend="rules")
    print(f"source={pipe.source.name} chunks={len(pipe.chunks)} embeddings={pipe.retriever.embedder.backend} "
          f"text-extraction={pipe.extraction.get('strategy')} {pipe.extraction.get('scores') or ''}\n")

    results: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in rows:
        if r["query"] not in results:
            recs, _, used = pipe.run(r["query"])
            results[r["query"]] = [public(x) for x in recs]
    passed = 0
    failed_queries = []
    for i, r in enumerate(rows, 1):
        got = results[r["query"]]
        if r["value"] == "NONE":
            ok = not got
        else:
            ok = any(r["component_contains"].lower() in (g["component"] + " " + g["spec_type"]).lower()
                     and norm(g["value"]) == norm(r["value"])
                     and g["unit"] == r["unit"] and r["section_contains"] in g["section"] for g in got)
        passed += ok
        mark = "PASS" if ok else "FAIL"
        want = "no result" if r["value"] == "NONE" else f'{r["component_contains"]} = {r["value"]} {r["unit"]} [{r["section_contains"]}]'
        print(f"{i:2d} {mark}  {r['query']}\n       expected: {want}")
        if not ok:
            failed_queries.append(r["query"])
            print(f"       got {len(got)} result(s):")
            for g in got[:6]:
                print(f"         - {g['component']} | {g['spec_type']} | {g['value']} {g['unit']} | {g['section']}")
    print(f"\nRESULT: {passed}/{len(rows)} expectations passed "
          f"({len(results) - len(set(failed_queries))}/{len(results)} queries fully correct)")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
