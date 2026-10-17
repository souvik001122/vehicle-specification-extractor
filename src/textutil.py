"""Small text helpers shared by retrieval and extraction: tokenizer, stemmer, query parser."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[a-z0-9]+")

# spelling variants that should match each other
_ALIAS = {
    "bleeder": "bleed", "bleeding": "bleed", "tightening": "tighten", "litre": "liter",
    "litres": "liter", "liters": "liter", "torques": "torque", "tightened": "tighten",
    "capacities": "capacity", "specs": "spec", "specifications": "specification",
}

# words that do not help to pick a component
STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "what", "whats", "is", "are", "value", "values", "spec", "specs",
    "specification", "specifications", "find", "get", "give", "me", "show", "list", "all", "and", "on", "in",
    "with", "vehicle", "manual", "required", "correct", "proper", "tighten", "torque", "nm", "lb", "ft",
    "capacity", "fill", "how", "much", "many", "does", "do", "need", "needs", "part", "number", "no", "size",
    "setting", "settings", "i", "my", "f", "150", "ford", "please", "tell", "about", "when", "installing",
    "install", "removal", "remove", "specified", "using", "use", "at", "by", "from", "or", "should", "be",
    "it", "its", "this", "that", "these", "those", "amount", "level", "volume", "refill", "numbers",
}

# words that tell which kind of specification is requested
_TYPE_WORDS = {
    "torque": {"torque", "tighten", "tightening", "nm", "lb-ft", "lb-in", "tight"},
    "capacity": {"capacity", "capacities", "fill", "refill", "volume", "liter", "liters", "litre", "litres",
                 "quart", "quarts", "ml", "pint", "pints", "much"},
    "measurement": {"thickness", "runout", "backlash", "clearance", "taper", "angle", "preload", "play",
                    "endplay", "diameter", "length", "wear", "breakaway", "variation", "width", "depth"},
}
_PART_RE = re.compile(r"\bpart\s*(?:number|no\.?|#)s?\b|\bp/n\b|\bpart\s+numbers\b", re.I)

# words that may be satisfied by the section title instead of the component name ("rear caliper bolts")
CONTEXT_WORDS = {"front", "rear", "left", "right", "lh", "rh", "4wd", "rwd", "4x4", "4x2", "raptor", "svt"}

FASTENERS = {"bolt", "nut", "screw", "stud", "plug", "fitting", "clamp", "washer"}

_SYNONYMS = {
    "fluid": {"fluid", "oil", "lubricant"},
    "oil": {"oil", "lubricant", "fluid"},
    "lubricant": {"lubricant", "oil", "fluid"},
}


def stem(word: str) -> str:
    """Very light plural stemmer (bolts -> bolt, axles -> axle, flanges -> flange)."""
    w = _ALIAS.get(word, word)
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens, stemmed. Hyphens split words (lb-ft -> lb, ft)."""
    return [stem(t) for t in _TOKEN.findall(text.lower())]


def synonyms(token: str) -> set[str]:
    return _SYNONYMS.get(token, {token})


@dataclass
class QuerySpec:
    raw: str
    spec_type: str | None = None          # torque | capacity | part_number | measurement | None (any)
    keywords: list[str] = field(default_factory=list)   # stemmed component words
    fastener: str | None = None           # bolt / nut / screw ... if the query names one


def parse_query(query: str) -> QuerySpec:
    """Work out which kind of spec is asked for and which component words must match."""
    q = query.strip()
    low = q.lower()
    raw_tokens = _TOKEN.findall(low)
    spec_type = None
    if _PART_RE.search(low):
        spec_type = "part_number"
    else:
        scores = {t: sum(1 for w in raw_tokens if w in words) + (1 if re.search(r"lb-?ft|lb-?in", low) and t == "torque" else 0)
                  for t, words in _TYPE_WORDS.items()}
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            spec_type = best
    keywords: list[str] = []
    for t in raw_tokens:
        if t in STOPWORDS:
            continue
        s = stem(t)
        if s not in keywords:
            keywords.append(s)
    fastener = next((k for k in keywords if k in FASTENERS), None)
    return QuerySpec(raw=q, spec_type=spec_type, keywords=keywords, fastener=fastener)


# query expansion for lexical + dense retrieval
_EXPANSION = {
    "torque": "torque tighten Nm lb-ft lb-in Torque Specifications",
    "capacity": "capacity fill Specification Material lubricant fluid ml L pt",
    "part_number": "Item Part Number Description",
    "measurement": "Specification mm in maximum minimum",
}


def expand_query(query: str, qs: QuerySpec | None = None) -> str:
    qs = qs or parse_query(query)
    extra = _EXPANSION.get(qs.spec_type or "", "")
    return f"{query} {extra}".strip()
