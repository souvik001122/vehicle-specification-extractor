"""Step 1b - rebuild table rows that a PDF text extractor split into one cell per line.

Depending on the PDF library, the same table can come out in different shapes::

    one line per row (text dump)        one line per cell (some PDF extractors)
    --------------------------------    ---------------------------------------
    Description Nm lb-ft lb-in          Description
    Wheel hub nut 40 30 —               Nm
                                        lb-ft
                                        lb-in
                                        Wheel hub nut
                                        40
                                        30
                                        —

The parsers in ``tables.py`` and the table detection in ``chunking.py`` expect the left-hand shape.
``reflow`` converts the right-hand shape into the left-hand one, for the three table kinds that
carry specifications (torque, specification, parts).  Text that is already in the one-line-per-row
shape passes through unchanged, so it is safe to run on every manual.
"""
from __future__ import annotations

import re

from .tables import DASHES, NUM, UNITS_T

_DASH = "—"
_UNIT_WORDS = r"(?:mm|cm|°|Nm|lb-ft|lb-in|kPa|psi|ml|mL|L|kg|degrees?|in)"
_VALUE_LINE = re.compile(
    r"^\d+(?:\.\d+)?(?:\s*°?\s*[-–]\s*\d+(?:\.\d+)?)?\s*" + _UNIT_WORDS + r"(?![A-Za-z])")
_ONE_LINE_SPEC = re.compile(
    r"^.*?[A-Za-z)].*?\s\d+(?:\.\d+)?(?:\s*°?\s*[-–]\s*\d+(?:\.\d+)?)?\s*" + _UNIT_WORDS + r"(?![A-Za-z])")
_PART_TOKEN = re.compile(r"^(?:[A-Z]{1,3}\d[\w-]*|\d[A-Z]\d*[\w-]*|\d{2,}[\w-]*|[—–-])$")
_ITEM_TOKEN = re.compile(r"^(?:\d{1,2}|[—–-])$")
_PROSE_START = re.compile(r"^(?:\d+\.\s|NOTE\b|NOTICE\b|WARNING\b|CAUTION\b|-\s|SECTION\b)")
_ONE_LINE_PART = re.compile(r"^(?:\d{1,2}|[—–-])\s+(?:[A-Z]{1,3}\d[\w-]*|\d[A-Z]\d*[\w-]*|\d{2,}[\w-]*|[—–-])\s+\S")
_MAX_FRAGMENTS = 3


def _is_cell(tok: str) -> bool:
    return bool(NUM.match(tok)) or tok in DASHES


def _all_cells(line: str) -> bool:
    toks = line.split()
    return bool(toks) and all(_is_cell(t) for t in toks)


def _prose(line: str) -> bool:
    return len(line) > 90 or line.endswith((".", ":")) or bool(_PROSE_START.match(line))


# --------------------------------------------------------------------------- headers
def _header_at(lines: list[str], i: int) -> tuple[str, str, int] | None:
    """(kind, canonical header line, lines used) when a *split* header starts at lines[i]."""
    n = len(lines)
    if lines[i] == "Description":
        units = []
        j = i + 1
        while j < n and len(units) < 3 and lines[j] in UNITS_T:
            units.append(lines[j])
            j += 1
        if units:
            return "torque", "Description " + " ".join(units), j - i
    if lines[i] == "Item":
        rest = []
        j = i + 1
        while j < n and j - i <= 4:
            rest.append(lines[j])
            joined = " ".join(rest)
            if joined == "Specification":
                return "spec", "Item Specification", j - i + 1
            if joined in ("Part Number Description", "Part Number", "Part"):
                if joined == "Part Number Description":
                    return "parts", "Item Part Number Description", j - i + 1
                if joined == "Part Number" and j + 1 < n and lines[j + 1] == "Description":
                    return "parts", "Item Part Number Description", j - i + 2
                if joined == "Part" and j + 2 < n and lines[j + 1] == "Number" and lines[j + 2] == "Description":
                    return "parts", "Item Part Number Description", j - i + 3
            if joined.startswith("Specification Fill"):
                if joined in ("Specification Fill Capacity",):
                    return "material", "Item Specification Fill Capacity", j - i + 1
            j += 1
    return None


def _any_header(lines: list[str], i: int) -> bool:
    from .tables import table_start
    return table_start(lines, i) is not None or _header_at(lines, i) is not None


# --------------------------------------------------------------------------- row builders
def _torque_rows(lines: list[str], i: int, ncols: int, out: list[str]) -> int:
    name: list[str] = []
    vals: list[str] = []
    n = len(lines)

    def flush_raw():
        out.extend(name)
        out.extend(vals)
        name.clear()
        vals.clear()

    while i < n:
        ln = lines[i]
        if _any_header(lines, i):
            break
        if _all_cells(ln) and name:
            vals.extend(ln.split())
            i += 1
            if len(vals) >= ncols:
                out.append(" ".join(name + vals[:ncols]))
                name.clear()
                vals.clear()
            continue
        toks = ln.split()
        if len(toks) > ncols and all(_is_cell(t) for t in toks[-ncols:]) and not _is_cell(toks[0]) and not name:
            out.append(ln)                      # already a one-line row
            i += 1
            continue
        if _prose(ln) or _all_cells(ln):
            break
        if vals or (name and not ln[:1].islower()):   # half-finished row / new row: keep earlier lines as they are
            flush_raw()
        name.append(ln)
        if len(name) > _MAX_FRAGMENTS:
            out.append(name.pop(0))
        i += 1
    flush_raw()
    return i


def _spec_rows(lines: list[str], i: int, out: list[str]) -> int:
    name: list[str] = []
    n = len(lines)
    while i < n:
        ln = lines[i]
        if _any_header(lines, i):
            break
        if name and _VALUE_LINE.match(ln):
            out.append(" ".join(name) + " " + ln)
            name.clear()
            i += 1
            continue
        if _ONE_LINE_SPEC.match(ln) and not _prose(ln) and not name:
            out.append(ln)
            i += 1
            continue
        if _prose(ln):
            break
        if name and not ln[:1].islower():       # a sub-heading such as "Brake Disc" is not part of the next row's name
            out.extend(name)
            name.clear()
        name.append(ln)
        if len(name) > _MAX_FRAGMENTS:
            out.append(name.pop(0))
        i += 1
    out.extend(name)
    return i


def _parts_rows(lines: list[str], i: int, out: list[str]) -> int:
    n = len(lines)
    item = part = None
    desc: list[str] = []

    def flush():
        nonlocal item, part
        if item is not None:
            out.append(" ".join([item, part or _DASH, *desc]).strip())
        item = part = None
        desc.clear()

    while i < n:
        ln = lines[i]
        if _any_header(lines, i):
            break
        if _ONE_LINE_PART.match(ln):
            flush()
            out.append(ln)
            i += 1
            continue
        if _ITEM_TOKEN.match(ln) and i + 1 < n and _PART_TOKEN.match(lines[i + 1]):
            flush()
            item, part = ln, lines[i + 1]
            i += 2
            continue
        # the first description line may be anything; later lines continue a wrapped cell only when they
        # start in lower case or with "(" - a Title-Case line such as "Installation" is the next heading
        if item is not None and not _prose(ln) and (not desc or ln[:1].islower() or ln.startswith("(")):
            desc.append(ln)
            i += 1
            continue
        break
    flush()
    return i


# --------------------------------------------------------------------------- entry point
def reflow(lines: list[str]) -> list[str]:
    """Return ``lines`` with split-cell tables rebuilt as one line per row."""
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        hdr = _header_at(lines, i)
        if hdr is None:
            out.append(lines[i])
            i += 1
            continue
        kind, text, used = hdr
        out.append(text)
        i += used
        if kind == "torque":
            i = _torque_rows(lines, i, len(text.split()) - 1, out)
        elif kind == "spec":
            i = _spec_rows(lines, i, out)
        elif kind == "parts":
            i = _parts_rows(lines, i, out)
    return out


def looks_flattened(lines: list[str]) -> bool:
    """True when at least one split table header is present (used for diagnostics)."""
    return any(_header_at(lines, i) is not None for i in range(len(lines)))
