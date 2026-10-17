"""Table detection and row parsing for the four table shapes used in the manual.

* torque table      ``Description Nm lb-ft lb-in`` + rows ``Brake caliper guide pin bolts 37 27 —``
* spec table        ``Item Specification`` + rows ``Brake disc minimum thickness 32 mm (1.259 in)``
* material table    ``Item Specification Fill Capacity`` (fluids, sealants; wrapped multi-line cells)
* parts table       ``Item Part Number Description`` + rows ``1 W506545 Shock absorber lower bolt``

The chunker uses the ``*_start`` helpers to find tables; the extractor uses the ``parse_*``
helpers to turn their lines into structured values.
"""
from __future__ import annotations

import re

DASHES = {"—", "–", "-", "‒"}
NUM = re.compile(r"^\d+(?:\.\d+)?$")
UNITS_T = ("Nm", "lb-ft", "lb-in")

TORQUE_HDR = re.compile(r"^Description\s+((?:Nm|lb-ft|lb-in)(?:\s+(?:Nm|lb-ft|lb-in))*)\s*$")
SPEC_HDR = re.compile(r"^Item\s+Specification\s*$")
MATERIAL_HDR = re.compile(r"^Item\s+Specification\s+Fill(?:\s+Capacity)?\s*$")
PARTS_HDR = re.compile(r"^Item\s+Part\s+Number\s+Description\s*$")
FOOTNOTE_RE = re.compile(r"^[a-z]\s+Refer\b", re.I)
SENTENCE_START = re.compile(r"^(?:\d+\.|-\s|NOTE\b|NOTICE\b|WARNING\b|CAUTION\b)")


# --------------------------------------------------------------------------- detection
def table_start(lines: list[str], i: int) -> tuple[str, int] | None:
    """Return (kind, header_line_count) when a table header starts at lines[i]."""
    ln = lines[i]
    if TORQUE_HDR.match(ln):
        return "torque_table", 1
    if MATERIAL_HDR.match(ln):
        # header is sometimes wrapped: 'Item Specification Fill' / 'Capacity'
        if ln.rstrip().endswith("Fill") and i + 1 < len(lines) and lines[i + 1].strip() == "Capacity":
            return "material_table", 2
        return "material_table", 1
    if SPEC_HDR.match(ln):
        return "spec_table", 1
    if PARTS_HDR.match(ln):
        return "parts_table", 1
    if ln == "Item" and i + 2 < len(lines) and lines[i + 1] == "Part" and lines[i + 2].startswith("Number Description"):
        return "parts_table", 3
    return None


def torque_columns(header: str) -> list[str]:
    m = TORQUE_HDR.match(header)
    return m.group(1).split() if m else []


# --------------------------------------------------------------------------- torque
def parse_torque_row(line: str, columns: list[str]) -> tuple[str, dict[str, str]] | None:
    """'Brake caliper guide pin bolts 37 27 —' -> ('Brake caliper guide pin bolts', {'Nm': '37', 'lb-ft': '27'})"""
    toks = line.split()
    vals: list[str] = []
    while toks and len(vals) < len(columns) and (NUM.match(toks[-1]) or toks[-1] in DASHES):
        vals.append(toks.pop())
    vals.reverse()
    if len(vals) != len(columns) or not toks or not any(NUM.match(v) for v in vals):
        return None
    name = " ".join(toks)
    name = re.sub(r"\s[a-z]$", "", name).strip()          # footnote marker 'a'
    if not re.search(r"[A-Za-z]{2}", name):
        return None
    return name, {c: v for c, v in zip(columns, vals) if NUM.match(v)}


def parse_torque_table(lines: list[str]) -> list[dict]:
    """Parse every torque row in ``lines`` (header line first). Handles wrapped names."""
    cols: list[str] = []
    out: list[dict] = []
    pending: list[str] = []
    for ln in lines:
        if TORQUE_HDR.match(ln):
            cols, pending = torque_columns(ln), []
            continue
        if not cols or FOOTNOTE_RE.match(ln) or ln.lower().endswith("specifications"):
            continue
        row = parse_torque_row(ln, cols)
        if row:
            name, vals = row
            if pending and name[:1].islower():          # wrapped name: previous fragment + this line
                name = " ".join(pending) + " " + name
            pending = []
            for unit in UNITS_T:                          # prefer metric, then lb-ft, then lb-in
                if unit in vals:
                    out.append({"component": name, "value": vals[unit], "unit": unit})
                    break
        elif len(ln) < 70 and not ln.endswith(".") and not SENTENCE_START.match(ln):
            pending = (pending + [ln])[-1:]
        else:
            pending = []
    return out


# --------------------------------------------------------------------------- measurements
_UNIT = r"(?:mm|cm|°|Nm|lb-ft|lb-in|kPa|psi|ml|mL|L|kg|degrees?|in)"
MEAS_RE = re.compile(
    r"^(?P<name>.*?[A-Za-z)].*?)\s+(?P<val>\d+(?:\.\d+)?(?:\s*°?\s*[-–]\s*\d+(?:\.\d+)?)?)\s*(?P<unit>" + _UNIT + r")(?![A-Za-z])"
)
QUALIFIERS = {"minimum", "maximum", "allowable", "nominal", "total", "min", "max"}
NOUNS = [  # longest first
    "rotational preload", "torque preload", "breakaway torque", "taper wear", "operating angle", "end play",
    "thickness", "runout", "backlash", "clearance", "preload", "variation", "diameter", "length", "width",
    "height", "depth", "angle", "pressure", "torque", "travel", "play", "distance", "wear", "gap", "spread",
    "level",
]
_LEAD_PREP = re.compile(r"^(?:between|of|for)\s+", re.I)


def split_spec_name(name: str) -> tuple[str, str]:
    """'Brake disc minimum thickness' -> ('Brake disc', 'Minimum thickness')."""
    note = ""
    pm = re.search(r"\s*(\([^)]*\))\s*$", name)
    if pm:
        note, name = " " + pm.group(1), name[:pm.start()]
    low = name.lower()
    found = []
    for n in NOUNS:
        if re.search(r"\b" + re.escape(n) + r"\b", low) and not any(n in f for f in found):
            found.append(n)
    if len(found) != 1:
        return name.strip() + note, "Measurement"
    noun = found[0]
    words = name.split()
    quals = [w for w in words if w.lower() in QUALIFIERS]
    noun_words = noun.split()
    rest, i = [], 0
    low_words = [w.lower() for w in words]
    while i < len(words):
        if low_words[i:i + len(noun_words)] == noun_words:
            i += len(noun_words)
            continue
        if low_words[i] in QUALIFIERS:
            i += 1
            continue
        rest.append(words[i])
        i += 1
    component = _LEAD_PREP.sub("", " ".join(rest)).strip() or name
    spec_type = " ".join([*(q.lower() for q in quals), noun]).strip()
    return component[:1].upper() + component[1:], spec_type[:1].upper() + spec_type[1:] + note


def parse_spec_table(lines: list[str]) -> list[dict]:
    """Single-line measurement rows of an 'Item Specification' table."""
    out = []
    for ln in lines:
        if SPEC_HDR.match(ln) or MATERIAL_HDR.match(ln) or TORQUE_HDR.match(ln):
            continue
        m = MEAS_RE.match(ln)
        if not m:
            continue
        name = m.group("name").strip()
        if re.search(r"\bin steps of$", name) or not re.search(r"[A-Za-z]{3}", name):
            continue
        val = re.sub(r"\s*°?\s*[-–]\s*", "-", m.group("val")).replace("°", "")
        component, spec_type = split_spec_name(name)
        out.append({"component": component, "spec_type": spec_type, "value": val, "unit": m.group("unit")})
    return out


# --------------------------------------------------------------------------- materials
CAP_RE = re.compile(r"(?<![\w.(])(\d[\d,]*(?:\.\d+)?)\s*(ml|mL|L|l|qt|pt|oz|gal)\b")
_CODE_TOKEN = re.compile(r"^(?:[A-Z]{1,5}[A-Z0-9]*-[A-Z0-9][A-Z0-9-]*|[A-Z]{1,2}\d?)$")
_CODE_WORDS = {"or", "and", "US", "Canada"}


def is_code_line(line: str) -> bool:
    """Part-number / specification-code fragments such as 'XY-75W140-QL (US); CXY75W140-1L' or '(Canada)'."""
    s = line.strip()
    if not s:
        return False
    if s.startswith("("):
        return True
    toks = s.split()
    has_code = False
    for t in toks:
        core = t.strip("();,")
        if core in _CODE_WORDS:
            continue
        if _CODE_TOKEN.match(core):
            has_code = True
            continue
        return False
    return has_code


def is_terminator(line: str) -> bool:
    s = line.strip()
    return bool(re.fullmatch(r"(?:[—–-]\s*)+", s)) or bool(re.search(r"\s[—–]$", s) and (is_code_line(s[:-1].strip()) or s.endswith("— —")))


def _clean_name(parts: list[str]) -> str:
    name = " ".join(p.strip() for p in parts)
    name = re.sub(r"[®™]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def parse_material_table(lines: list[str]) -> list[dict]:
    """Return one dict per material entry that has a fill capacity.

    The cells of this table are wrapped over several lines, so the entry name is recovered by
    walking backwards from the capacity line: first over the code lines (part number / Ford
    specification), then over the name lines, stopping at the previous entry.
    Keys: name, part_numbers, spec_codes, capacities=[{value, unit, qualifier}].
    """
    body = [ln for ln in lines if not (MATERIAL_HDR.match(ln) or SPEC_HDR.match(ln) or ln == "Capacity")]
    out: list[dict] = []
    n = len(body)
    i = 0
    while i < n:
        if not _primary_capacity(body[i]):
            i += 1
            continue
        group, j = _capacity_group(body, i)
        names, codes = _entry_above(body, i)
        name = _clean_name(names)
        if name:
            part_numbers, spec_codes = _split_codes(codes)
            if len(group) == 1:
                group[0]["qualifier"] = ""            # a single capacity: trailing text is a note, not a variant
            for g in group:
                g.pop("_dash", None)
            out.append({"name": name, "part_numbers": part_numbers, "spec_codes": spec_codes, "capacities": group})
        i = max(j, i + 1)
    return out


_DASH_TAIL = re.compile(r"[—–]\s*(.*)$")


def _primary_capacity(line: str) -> tuple[str, str] | None:
    """First capacity not inside parentheses: '2.84L (6.0 pt) — ELD only' -> ('2.84', 'L')."""
    if is_code_line(line):
        return None
    m = CAP_RE.search(line)
    if not m:
        return None
    unit = "L" if m.group(2) == "l" else m.group(2)
    return m.group(1).replace(",", ""), unit


def _capacity_group(body: list[str], i: int) -> tuple[list[dict], int]:
    """Consecutive capacity lines of one entry, e.g. 'ELD only' / 'Non Traction-Lok' / 'Traction-Lok'."""
    group: list[dict] = []
    n = len(body)
    j = i
    while j < n:
        cap = _primary_capacity(body[j])
        if not cap:
            break
        m = _DASH_TAIL.search(body[j])
        group.append({"value": cap[0], "unit": cap[1], "qualifier": m.group(1).strip() if m else "", "_dash": bool(m)})
        j += 1
        nxt = body[j] if j < n else ""
        if nxt and not _primary_capacity(nxt) and not is_code_line(nxt) and not is_terminator(nxt):
            g = group[-1]
            if nxt.startswith("("):                                   # imperial amount '(3 pt)'
                j += 1
            elif g["_dash"] and (len(g["qualifier"].split()) <= 1 or nxt[:1].islower()):
                g["qualifier"] = (g["qualifier"] + " " + nxt).strip()  # wrapped qualifier
                j += 1
            elif nxt[:1].islower():                                   # wrapped note
                j += 1
    return group, j


def _entry_above(body: list[str], i: int) -> tuple[list[str], list[str]]:
    """(name lines, code lines) of the entry whose capacity is on body[i]."""
    codes: list[str] = []
    m = CAP_RE.search(body[i])
    prefix = body[i][:m.start()].strip() if m else ""
    if prefix and is_code_line(prefix):
        codes.append(prefix)
    k = i - 1
    while k >= 0 and is_code_line(body[k]):
        codes.insert(0, body[k])
        k -= 1
    moved: list[str] = []
    while codes and re.fullmatch(r"\((?:US|Canada)\)", codes[0].strip()):   # region label of the *name*
        moved.append(codes.pop(0))
    names: list[str] = []
    while k >= 0 and len(names) < 6:
        ln = body[k]
        if is_terminator(ln) or _primary_capacity(ln) or is_code_line(ln) or ln[:1].islower():
            break
        names.insert(0, ln)
        k -= 1
    return names + moved, codes


_SPEC_TOKEN = re.compile(r"^(?:WS[A-Z]{0,2}|ES[A-Z]{1,2})-")


def _split_codes(codes: list[str]) -> tuple[str, str]:
    """Separate Motorcraft part numbers (XY-75W140-QL) from Ford material specifications (WSL-M2C192-A)."""
    joined = ""
    for c in codes:                                   # re-join cells wrapped after a hyphen
        c = c.strip()
        joined = joined + c if joined.endswith("-") else (joined + " " + c).strip()
    toks = joined.split()
    idx = next((k for k, t in enumerate(toks) if _SPEC_TOKEN.match(t) and not (k and toks[k - 1] == "or" and k > 1 and not _SPEC_TOKEN.match(toks[k - 2]))), len(toks))
    if idx and toks[idx - 1] == "or":
        idx -= 1
    return " ".join(toks[:idx]), " ".join(toks[idx:])


# --------------------------------------------------------------------------- parts
PART_ROW = re.compile(r"^(?P<item>\d+)\s+(?P<pn>[A-Z0-9][A-Z0-9-]*|[—–-])\s+(?P<desc>\S.*)$")


def consume_parts(lines: list[str], start: int) -> int:
    """Index just after the parts-table rows that begin at ``start`` (header already skipped)."""
    i = start
    n = len(lines)
    seen_row = False
    while i < n:
        ln = lines[i]
        if PART_ROW.match(ln) and not re.match(r"^\d+\.\s", ln):
            seen_row = True
            i += 1
            continue
        if seen_row and not SENTENCE_START.match(ln) and not ln.endswith(".") and len(ln) < 70 \
                and table_start(lines, i) is None and i > start and PART_ROW.match(lines[i - 1]) \
                and ln[:1] in "([abcdefghijklmnopqrstuvwxyz":
            i += 1                                   # wrapped description
            continue
        break
    return i


def parse_parts_table(lines: list[str]) -> list[dict]:
    out: list[dict] = []
    for ln in lines:
        m = PART_ROW.match(ln)
        if m and not re.match(r"^\d+\.\s", ln):
            out.append({"item": m.group("item"), "part_number": m.group("pn"), "component": m.group("desc").strip()})
        elif out and not TORQUE_HDR.match(ln) and ln[:1] in "([abcdefghijklmnopqrstuvwxyz":
            out[-1]["component"] += " " + ln.strip()
    return out
