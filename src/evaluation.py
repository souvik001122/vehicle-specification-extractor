"""Batch evaluation: run many questions and compare the answers with expected ones.

Used by the Streamlit "Batch check" tab and by ``tests/evaluate.py``, so both always agree.

CSV columns (only ``query`` is required)::

    query, component_contains, value, unit, section_contains, manual_line

* a row with ``value`` is a check: PASS when the pipeline returns a result whose component (or spec type)
  contains ``component_contains``, with the same value and unit, and a section that contains
  ``section_contains``;
* ``value`` = ``NONE`` means the manual has no answer, so the pipeline must return nothing;
* a row without ``value`` is just run (status ``RAN``) and its results are listed.
"""
from __future__ import annotations

import csv
import io
from typing import Iterable

from .extractor import public

PASS, FAIL, RAN = "PASS", "FAIL", "RAN"


def load_expectations(source: str | bytes) -> list[dict]:
    """Parse CSV text/bytes into a list of row dicts. Raises ValueError if there is no ``query`` column."""
    text = source.decode("utf-8-sig") if isinstance(source, bytes) else source.lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text))
    fields = [f.strip().lower() for f in (reader.fieldnames or [])]
    if "query" not in fields:
        raise ValueError("The CSV needs a 'query' column (optional: component_contains, value, unit, "
                         "section_contains, manual_line).")
    rows = []
    for raw in reader:
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        if row.get("query"):
            rows.append(row)
    return rows


def _norm(v: str) -> str:
    return v.replace(",", "").strip()


def check_row(row: dict, got: list[dict]) -> str:
    """PASS / FAIL / RAN for one expectation row against the public result dicts ``got``."""
    want = row.get("value", "")
    if not want:
        return RAN
    if want.upper() == "NONE":
        return PASS if not got else FAIL
    comp = row.get("component_contains", "").lower()
    sec = row.get("section_contains", "")
    unit = row.get("unit", "")
    for g in got:
        if (comp in (g["component"] + " " + g["spec_type"]).lower() and _norm(g["value"]) == _norm(want)
                and g["unit"] == unit and sec in g.get("section", "")):
            return PASS
    return FAIL


def describe(row: dict) -> str:
    want = row.get("value", "")
    if not want:
        return ""
    if want.upper() == "NONE":
        return "no result"
    return f'{row.get("component_contains", "")} = {want} {row.get("unit", "")} [{row.get("section_contains", "")}]'.replace("  ", " ")


def run_batch(pipe, rows: Iterable[dict], top_k: int | None = None, progress=None) -> list[dict]:
    """Run every distinct query once and grade every row. Returns one dict per row."""
    rows = list(rows)
    cache: dict[str, list[dict]] = {}
    out = []
    for i, row in enumerate(rows):
        q = row["query"]
        if q not in cache:
            recs, _, _ = pipe.run(q, top_k)
            cache[q] = [public(r) for r in recs]
        got = cache[q]
        status = check_row(row, got)
        out.append({
            "status": status,
            "query": q,
            "expected": describe(row),
            "results": len(got),
            "got": " | ".join(f"{g['component']} = {g['value']} {g['unit']} [{g['section'].split(':')[0]}]"
                              for g in got[:6]) + (" ..." if len(got) > 6 else ""),
        })
        if progress:
            progress((i + 1) / len(rows))
    return out


def summarize(report: list[dict]) -> dict:
    checks = [r for r in report if r["status"] in (PASS, FAIL)]
    passed = sum(1 for r in checks if r["status"] == PASS)
    queries = {r["query"] for r in checks}
    bad = {r["query"] for r in checks if r["status"] == FAIL}
    return {"checks": len(checks), "passed": passed, "failed": len(checks) - passed,
            "queries": len(queries), "queries_fully_correct": len(queries) - len(bad),
            "run_only": sum(1 for r in report if r["status"] == RAN)}
