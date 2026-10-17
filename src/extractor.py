"""Step 5 - structured extraction from the retrieved chunks.

Two interchangeable back ends produce the same ``Record`` objects:

* **LLM** (OpenAI or Anthropic): the retrieved chunks and the question go into a strict-JSON
  prompt.  Every returned value is checked against the retrieved text (hallucination guard).
* **Rules** (offline, no key): the table parsers in ``tables.py`` plus regexes for
  ``Tighten to 37 Nm (27 lb-ft)`` style steps and ``fill with 2.8L`` capacities.  Candidate rows
  are then filtered and ranked against the component words of the question.

``auto`` uses an LLM when an API key is present and falls back to the rules if the LLM fails or
returns nothing.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import config
from .chunking import Chunk
from .tables import (
    UNITS_T, parse_material_table, parse_parts_table, parse_spec_table, parse_torque_table,
)
from .textutil import CONTEXT_WORDS, FASTENERS, QuerySpec, parse_query, synonyms, tokenize

log = logging.getLogger(__name__)

TORQUE_UNITS = set(UNITS_T)


@dataclass
class Record:
    component: str
    spec_type: str
    value: str
    unit: str = ""
    section: str = ""
    source: str = "table"            # table | procedure | llm
    chunk_id: int = -1
    section_id: str = field(default="", repr=False)

    @property
    def key(self) -> tuple:
        return (self.component.lower(), self.spec_type.lower(), self.value, self.unit, self.section_id)


def public(r: "Record | dict", minimal: bool = False) -> dict:
    """The JSON/CSV view of a record. ``minimal`` gives exactly the four fields of the assignment."""
    d = r if isinstance(r, dict) else {
        "component": r.component, "spec_type": r.spec_type, "value": r.value, "unit": r.unit, "section": r.section,
    }
    out = {k: d.get(k, "") for k in ("component", "spec_type", "value", "unit")}
    if not minimal:
        out["section"] = d.get("section", "")
    return out


# =========================================================================== rule based
_TIGHTEN = re.compile(
    r"(?i)\btighten\b(?P<mid>[^.\n]{0,70}?)\bto\s+(?P<v>\d+(?:\.\d+)?)\s*Nm\s*\((?P<alt>[^)]*)\)"
)
_FASTENER_PHRASE = re.compile(
    r"(?i)(?:remove|install|loosen|discard|position|disconnect|connect|torque|tighten)\s+(?:and\s+discard\s+)?(?:the\s+)?"
    r"(?:\d+(?:\s+or\s+\d+)?\s+)?(?:new\s+)?(?P<c>[A-Za-z][A-Za-z0-9\-/®() ]{2,60}?(?:bolts?|nuts?|screws?|studs?|fittings?|plugs?))\b"
)
_FILL = re.compile(
    r"(?i)\bfill\b[^.\n]{0,60}?\bwith\s+(?P<v>\d[\d,]*(?:\.\d+)?)\s*(?P<u>ml|L)\b[^.;\n]*?\bof\s+"
    r"(?P<c>[A-Za-z][A-Za-z \-]{2,40}?)(?=[.;,]|\s+and\b|\s+to\b|$)"
)


def _section_of(chunk: Chunk) -> str:
    return chunk.section


def table_records(chunk: Chunk) -> list[Record]:
    """All records that can be read from a table chunk."""
    lines = chunk.text.split("\n")
    sec, sid = chunk.section, chunk.section_id
    out: list[Record] = []
    if chunk.kind == "torque_table":
        for r in parse_torque_table(lines):
            out.append(Record(_cap(r["component"]), "Torque", r["value"], r["unit"], sec, "table", chunk.id, sid))
    elif chunk.kind == "spec_table":
        for r in parse_spec_table(lines):
            out.append(Record(r["component"], r["spec_type"], r["value"], r["unit"], sec, "table", chunk.id, sid))
    elif chunk.kind == "material_table":
        for e in parse_material_table(lines):
            multi = len(e["capacities"]) > 1
            for c in e["capacities"]:
                name = e["name"] + (f" ({c['qualifier']})" if multi and c["qualifier"] else "")
                out.append(Record(name, "Fluid Capacity", c["value"], c["unit"], sec, "table", chunk.id, sid))
            if e["part_numbers"]:
                out.append(Record(e["name"], "Part Number", e["part_numbers"], "", sec, "table", chunk.id, sid))
            if e["spec_codes"]:
                out.append(Record(e["name"], "Material Specification", e["spec_codes"], "", sec, "table", chunk.id, sid))
    elif chunk.kind == "parts_table":
        for r in parse_parts_table(lines):
            if r["part_number"] in {"—", "–", "-"}:
                continue
            out.append(Record(_cap(r["component"]), "Part Number", r["part_number"], "", sec, "table", chunk.id, sid))
    return out


def procedure_records(chunk: Chunk, known: set[tuple]) -> list[Record]:
    """Torque / fill values written inside procedure steps that are not already in a spec table."""
    if chunk.kind != "text":
        return []
    text = re.sub(r"\s*\n\s*", " ", chunk.text)
    out: list[Record] = []
    heading = chunk.heading.split(" — ")[0].strip() or chunk.section_title
    for m in _TIGHTEN.finditer(text):
        unit = "Nm"
        value = m.group("v")
        if (chunk.section_id, value, unit) in known:
            continue
        window = text[max(0, m.start() - 260):m.start()]
        found = list(_FASTENER_PHRASE.finditer(window))
        component = _cap(found[-1].group("c").strip()) if found else f"{heading} fasteners"
        out.append(Record(component, "Torque", value, unit, chunk.section, "procedure", chunk.id, chunk.section_id))
    for m in _FILL.finditer(text):
        value = m.group("v").replace(",", "")
        out.append(Record(_cap(m.group("c").strip()), "Fluid Capacity", value, m.group("u"), chunk.section,
                          "procedure", chunk.id, chunk.section_id))
    return out


def _cap(s: str) -> str:
    s = re.sub(r"[®™]", "", s).strip()
    return s[:1].upper() + s[1:]


def _match_score(rec: Record, qs: QuerySpec) -> tuple[int, int]:
    """(keywords found, keywords found inside the component/spec_type itself).

    A keyword must be in the component (or spec_type) unless it is a region word such as
    front/rear/4WD, which may also be satisfied by the section title.
    """
    comp = set(tokenize(rec.component + " " + rec.spec_type))
    ctx = set(tokenize(rec.section))
    found = in_comp = 0
    for kw in qs.keywords:
        syn = synonyms(kw)
        if syn & comp:
            found += 1
            in_comp += 1
        elif kw in CONTEXT_WORDS and syn & ctx:
            found += 1
    return found, in_comp


def _type_ok(rec: Record, qs: QuerySpec) -> bool:
    t = qs.spec_type
    if t is None:
        return True
    if t == "torque":
        return rec.unit in TORQUE_UNITS
    if t == "capacity":
        return rec.spec_type == "Fluid Capacity"
    if t == "part_number":
        return rec.spec_type in {"Part Number"}
    if t == "measurement":
        return rec.spec_type not in {"Torque", "Fluid Capacity", "Part Number", "Material Specification"}
    return True


def filter_and_rank(records: list[Record], qs: QuerySpec) -> list[Record]:
    """Keep rows of the requested type whose component words match the question; best first."""
    rows = [r for r in records if _type_ok(r, qs)]
    if not qs.keywords:
        return rows
    n = len(qs.keywords)
    scored = [(r, *_match_score(r, qs)) for r in rows]
    strict = [(r, f, c) for r, f, c in scored if f == n and c >= 1]
    if qs.fastener:           # "caliper bolts" must not return bleed screws
        fast = [(r, f, c) for r, f, c in strict if FASTENERS & set(tokenize(r.component)) and
                any(syn in tokenize(r.component) for syn in synonyms(qs.fastener))]
        strict = fast or strict
    if not strict and n >= 3:  # relax: a long question may have one word the manual phrases differently
        strict = [(r, f, c) for r, f, c in scored if f >= n - 1 and c >= 2]
    strict.sort(key=lambda t: (-t[1], -t[2]))
    return [r for r, _, _ in strict]


def dedupe(records: list[Record]) -> list[Record]:
    seen, out = set(), []
    for r in records:
        if r.key not in seen:
            seen.add(r.key)
            out.append(r)
    return out


class RuleExtractor:
    """Offline extractor. ``index_corpus`` lets it recognise procedure values already covered by a table."""

    def __init__(self):
        self.known: set[tuple] = set()
        self.cap_sections: set[str] = set()
        self._cache: dict[int, list[Record]] = {}

    def index_corpus(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            if c.is_table:
                recs = self._table(c)
                for r in recs:
                    self.known.add((r.section_id, r.value, r.unit))
                    if r.spec_type == "Fluid Capacity":
                        self.cap_sections.add(r.section_id)

    def _table(self, chunk: Chunk) -> list[Record]:
        if chunk.id not in self._cache:
            self._cache[chunk.id] = table_records(chunk)
        return self._cache[chunk.id]

    def extract(self, query: str, hits: list[tuple[Chunk, float]]) -> list[Record]:
        qs = parse_query(query)
        cands: list[Record] = []
        for chunk, _ in hits:
            if chunk.is_table:
                cands.extend(self._table(chunk))
            else:
                for r in procedure_records(chunk, self.known):
                    if r.spec_type == "Fluid Capacity" and r.section_id in self.cap_sections:
                        continue
                    cands.append(r)
        return dedupe(filter_and_rank(cands, qs))


# =========================================================================== LLM based
SYSTEM_PROMPT = """You extract vehicle specifications from automotive service-manual excerpts.
Return ONLY a JSON object: {"specs": [{"component": str, "spec_type": str, "value": str, "unit": str, "chunk": int}]}

Rules:
- Use only the excerpts. Never invent or convert numbers. If nothing answers the question return {"specs": []}.
- One entry per component and value. Front/rear/variant rows (e.g. 4x4, SVT Raptor, ELD) are separate entries and the
  variant stays in the component name.
- component: the part as named in the manual, in sentence case (e.g. "Brake caliper guide pin bolts").
- spec_type: "Torque", "Fluid Capacity", "Part Number", "Minimum thickness", "Runout", ... as appropriate.
- value: the number (or range such as "0.203-0.304", or a part number) exactly as printed, without the unit and
  without thousands separators.
- unit: Nm, L, ml, mm ... For torque use the metric Nm column. Use "" when there is no unit (part numbers).
- chunk: the number of the excerpt the value came from.
- Ignore values that belong to other components, even if they look similar."""


def build_prompt(query: str, hits: list[tuple[Chunk, float]], max_chars: int = 24000) -> str:
    parts, used = [], 0
    for i, (c, _) in enumerate(hits, 1):
        block = f"[{i}] Section {c.section} | {c.mode} | {c.heading}\n{c.text}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return f"Question: {query}\n\nExcerpts:\n\n" + "\n".join(parts)


def _norm_num(s: str) -> str:
    return s.replace(",", "")


def value_in_text(value: str, text: str) -> bool:
    """Hallucination guard: the number (or each end of a range, or the part number) must be in the text."""
    value = value.strip()
    if not value:
        return False
    flat = re.sub(r"\s+", " ", text)
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?", _norm_num(value)):
        flat = _norm_num(flat)
        nums = re.split(r"\s*[-–]\s*", _norm_num(value))
        return all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?!\d)", flat) for n in nums)
    return re.sub(r"\s+", "", value).lower() in re.sub(r"\s+", "", flat).lower()


def parse_llm_json(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict):
        data = data.get("specs", data.get("results", []))
    return [d for d in (data or []) if isinstance(d, dict)]


def records_from_llm(items: list[dict], hits: list[tuple[Chunk, float]]) -> list[Record]:
    recs: list[Record] = []
    for d in items:
        value = _norm_num(str(d.get("value", "")).strip())
        comp = str(d.get("component", "")).strip()
        if not value or not comp:
            continue
        chunk = None
        try:
            idx = int(d.get("chunk", 0))
            if 1 <= idx <= len(hits) and value_in_text(value, hits[idx - 1][0].text):
                chunk = hits[idx - 1][0]
        except (TypeError, ValueError):
            pass
        if chunk is None:                                    # wrong/missing chunk number: find the value
            chunk = next((c for c, _ in hits if value_in_text(value, c.text)), None)
        if chunk is None:
            log.info("dropped LLM value not found in the retrieved text: %s %s", comp, value)
            continue
        recs.append(Record(comp, str(d.get("spec_type", "")).strip() or "Specification", value,
                           str(d.get("unit", "")).strip(), chunk.section, "llm", chunk.id, chunk.section_id))
    return dedupe(recs)


def _openai_call(prompt: str) -> str:
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=config.OPENAI_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def _anthropic_call(prompt: str) -> str:
    import anthropic
    resp = anthropic.Anthropic().messages.create(
        model=config.ANTHROPIC_MODEL, max_tokens=3000, temperature=0, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


# =========================================================================== facade
class Extractor:
    """Picks the back end and runs it. ``extract`` returns (records, backend_used)."""

    def __init__(self, backend: str | None = None, llm_fn: Callable[[str], str] | None = None):
        self.requested = (backend or config.EXTRACT_BACKEND).lower()
        self.rules = RuleExtractor()
        self.llm_fn = llm_fn
        self.llm_name = ""
        self._resolve()

    def _resolve(self) -> None:
        req = self.requested
        if self.llm_fn is not None:
            self.llm_name, self.name = "custom", "custom-llm"
            return
        if req == "rules":
            self.name = "rules"
            return
        order = {"auto": ["openai", "anthropic"], "openai": ["openai"], "anthropic": ["anthropic"]}.get(req)
        if order is None:
            raise ValueError(f"Unknown extract backend: {req}")
        for b in order:
            if b == "openai" and os.getenv("OPENAI_API_KEY") and _importable("openai"):
                self.llm_fn, self.llm_name = _openai_call, "openai"
                self.name = f"openai:{config.OPENAI_MODEL}"
                return
            if b == "anthropic" and os.getenv("ANTHROPIC_API_KEY") and _importable("anthropic"):
                self.llm_fn, self.llm_name = _anthropic_call, "anthropic"
                self.name = f"anthropic:{config.ANTHROPIC_MODEL}"
                return
        if req != "auto":
            raise RuntimeError(f"--extract-backend {req} needs its API key and python package installed")
        self.name = "rules"

    def index_corpus(self, chunks: list[Chunk]) -> None:
        self.rules.index_corpus(chunks)

    def extract(self, query: str, hits: list[tuple[Chunk, float]]) -> tuple[list[Record], str]:
        if self.llm_fn is not None:
            try:
                raw = self.llm_fn(build_prompt(query, hits))
                recs = records_from_llm(parse_llm_json(raw), hits)
                if recs:
                    return recs, self.llm_name
                log.info("LLM returned nothing usable - falling back to rules")
            except Exception as e:
                log.warning("LLM extraction failed (%s) - falling back to rules", e)
            return self.rules.extract(query, hits), "rules (LLM fallback)"
        return self.rules.extract(query, hits), "rules"


def _importable(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False
