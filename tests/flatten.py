"""Test helper: turn the manual's one-line-per-row tables into one-cell-per-line tables.

This imitates what some PDF text extractors do, so the tests can prove that ``src/reflow.py``
restores the same specifications.  Only used by the tests / for debugging.
"""
import re

_CELL = r"(?:\d+(?:\.\d+)?|—)"
_TORQUE_ROW = re.compile(rf"^(?P<name>.*?[A-Za-z)].*?)\s+(?P<v>{_CELL}(?:\s+{_CELL}){{1,2}})\s*$")
_UNIT = r"(?:mm|cm|°|Nm|lb-ft|lb-in|kPa|psi|ml|mL|L|kg|degrees?|in)"
_SPEC_ROW = re.compile(rf"^(?P<name>.*?[A-Za-z)].*?)\s+(?P<val>\d+(?:\.\d+)?(?:\s*°?\s*[-–]\s*\d+(?:\.\d+)?)?\s*{_UNIT}(?![A-Za-z]).*)$")
_PART_ROW = re.compile(r"^(?P<item>\d{1,2}|—)\s+(?P<part>(?:[A-Z]{1,3}\d[\w-]*|\d[A-Z]\d*[\w-]*|\d{2,}[\w-]*|—))\s+(?P<desc>\S.*)$")


def flatten(raw: str) -> str:
    out, mode = [], None
    for ln in raw.split("\n"):
        s = ln.strip()
        if re.match(r"^Description\s+(?:Nm|lb-ft|lb-in)", s):
            out += s.split(); mode = "torque"; continue
        if s == "Item Specification":
            out += s.split(); mode = "spec"; continue
        if s == "Item Part Number Description":
            out += ["Item", "Part Number", "Description"]; mode = "parts"; continue
        if mode == "torque":
            m = _TORQUE_ROW.match(s)
            if m and len(s) < 90:
                out.append(m.group("name")); out += m.group("v").split(); continue
        elif mode == "spec":
            m = _SPEC_ROW.match(s)
            if m and len(s) < 110 and not s.endswith("."):
                out.append(m.group("name")); out.append(m.group("val")); continue
        elif mode == "parts":
            m = _PART_ROW.match(s)
            if m:
                out += [m.group("item"), m.group("part"), m.group("desc")]; continue
        out.append(ln)
    return "\n".join(out)
