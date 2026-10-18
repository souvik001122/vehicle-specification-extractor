"""Step 1 - text extraction and cleaning.

Reads a service manual (.pdf, or a .txt dump of one) and turns it into structured
``Doc`` objects.  The manual is a print-out of many small HTML pages.  Every printed page
ends with a footer ``<vehicle> Workshop Manual Page X sur Y``; a *document* is the run of
pages from X=1 to X=Y.  Each document carries one ``SECTION 206-03: Front Disc Brake ...``
header plus a mode line (``SPECIFICATIONS``, ``REMOVAL AND INSTALLATION`` ...).  The text
extractor puts that header at an arbitrary place inside the document, so we pull it out and
apply it to the whole document instead of trusting its position.

Cleaning removes: footers, ``file:///`` print lines, revision dates, form feeds, the
Wingdings bullet ``z`` (becomes ``-``) and blank lines.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import config
from .reflow import reflow

log = logging.getLogger(__name__)

FOOTER_RE = re.compile(r"^\s*.*?\bPage\s+(\d+)\s+(?:sur|of)\s+(\d+)\s*$", re.I)
FILE_RE = re.compile(r"^\s*file:///", re.I)
HEADER_RE = re.compile(r"^SECTION\s+(\d{3}-\d{2}[A-Z]?)\s*:\s*(.+?)\s*$")
VEHICLE_TAIL_RE = re.compile(r"\s+(\d{4}\s+\S.*?Workshop Manual)\s*$", re.I)
MODE_RE = re.compile(r"^([A-Z][A-Z &/,\-]+?)\s+Procedure revision date:.*$")
BULLET_RE = re.compile(r"^[z•●\x84]\s+")
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
STRAY_RE = re.compile(r"^\s*Procedure revision date:.*$", re.I)


@dataclass
class Doc:
    """One manual page-set (a single procedure, specification page, description ...)."""
    index: int
    section_id: str = ""
    section_title: str = ""
    mode: str = ""
    vehicle: str = ""
    title: str = ""
    lines: list[str] = field(default_factory=list)
    pages: list[str] = field(default_factory=list)

    @property
    def section(self) -> str:
        return f"{self.section_id}: {self.section_title}" if self.section_id else self.section_title

    @property
    def is_spec(self) -> bool:
        return self.mode.upper().startswith("SPECIFICATION")


# --------------------------------------------------------------------------- reading
def read_raw(path: str | Path) -> str:
    """Return the raw text of a PDF or text file."""
    return read_raw_info(path)[0]


def read_raw_info(path: str | Path) -> tuple[str, dict]:
    """Return (raw text, info). ``info`` says which extraction strategy was used and how each one scored."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manual not found: {p}")
    if p.suffix.lower() in {".txt", ".text", ".md"}:
        return p.read_text(encoding="utf-8", errors="replace"), {"strategy": "text file", "scores": {}}
    return _read_pdf(p)


def rows_from_words(words, y_tol: float = 0.5) -> list[str]:
    """Rebuild visual text rows from PyMuPDF ``get_text("words")`` tuples ``(x0, y0, x1, y1, text, ...)``.

    Words whose vertical centres are close belong to one row, so a table row such as
    ``Brake caliper guide pin bolts | 37 | 27 | -`` comes out as one line, whatever order the PDF
    stores the cells in.
    """
    ws = [w for w in words if str(w[4]).strip()]
    ws.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list] = []
    cur: list = []
    cur_y = cur_h = 0.0
    for w in ws:
        yc, h = (w[1] + w[3]) / 2, max(w[3] - w[1], 1.0)
        if cur and abs(yc - cur_y) <= y_tol * min(h, cur_h):
            cur.append(w)
            cur_y += (yc - cur_y) / len(cur)
            cur_h = min(cur_h, h)
        else:
            if cur:
                rows.append(cur)
            cur, cur_y, cur_h = [w], yc, h
    if cur:
        rows.append(cur)
    return [" ".join(str(w[4]) for w in sorted(r, key=lambda w: w[0])) for r in rows]


def _pymupdf_pages(p: Path, mode: str) -> str:
    import fitz  # PyMuPDF
    out = []
    with fitz.open(p) as pdf:
        for page in pdf:
            if mode == "rows":
                text = "\n".join(rows_from_words(page.get_text("words")))
            else:
                text = page.get_text("text")
            if len(text.strip()) < 20:
                text = _ocr_page(page) or text
            out.append(text)
    return "\n".join(out)


def _pypdf_text(p: Path) -> str:
    from pypdf import PdfReader
    return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)


def _pdfminer_text(p: Path) -> str:
    from pdfminer.high_level import extract_text
    return extract_text(str(p))


STRATEGIES = [
    ("pymupdf-rows", lambda p: _pymupdf_pages(p, "rows")),
    ("pymupdf-text", lambda p: _pymupdf_pages(p, "text")),
    ("pypdf", _pypdf_text),
    ("pdfminer", _pdfminer_text),
]
GOOD_ENOUGH = 40          # specification rows found; below this the slower fallback libraries are tried too


def spec_yield(raw: str) -> int:
    """How many specification rows can be read from ``raw`` - the quality score of an extraction strategy."""
    from .chunking import chunk_documents
    from .extractor import table_records
    chunks = chunk_documents(parse_manual(raw))
    return sum(len(table_records(c)) for c in chunks if c.is_table)


def _read_pdf(p: Path) -> tuple[str, dict]:
    """Run several extraction strategies and keep the one that yields the most specification rows.

    PDF libraries lay tables out differently (one line per row, or one line per cell), and which
    one works best depends on how the PDF was produced - so the choice is made from the results.
    """
    import hashlib
    key = hashlib.sha1(p.read_bytes()).hexdigest()[:16]
    cached = config.CACHE_DIR / f"pdftext-{key}.txt"
    meta = config.CACHE_DIR / f"pdftext-{key}.json"
    if cached.exists() and meta.exists():
        try:
            import json
            return cached.read_text(encoding="utf-8"), {**json.loads(meta.read_text()), "cached": True}
        except Exception:
            pass
    best_name, best_raw, best_score, scores, errors = "", "", -1, {}, {}
    for name, fn in STRATEGIES:
        if best_score >= GOOD_ENOUGH and name in ("pypdf", "pdfminer"):
            break                                  # the fast PyMuPDF strategies were good enough
        try:
            raw = fn(p)
            score = spec_yield(raw)
        except Exception as e:                     # missing library, broken page ... try the next strategy
            errors[name] = f"{type(e).__name__}: {e}"
            log.info("PDF strategy %s failed: %s", name, e)
            continue
        scores[name] = score
        log.info("PDF strategy %s: %d specification rows", name, score)
        if score > best_score:
            best_name, best_raw, best_score = name, raw, score
    if not best_name:
        raise RuntimeError("No PDF library could read the file (" + "; ".join(f"{k}: {v}" for k, v in errors.items())
                           + "). pip install pymupdf")
    info = {"strategy": best_name, "scores": scores, "errors": errors, "cached": False}
    try:
        import json
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(best_raw, encoding="utf-8")
        meta.write_text(json.dumps({k: v for k, v in info.items() if k != "cached"}))
    except OSError:
        pass
    return best_raw, info


def _ocr_page(page) -> str:
    """Optional OCR for scanned pages (needs pytesseract + pillow + the tesseract binary)."""
    try:
        import io
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=200)
        return pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))))
    except Exception as e:  # OCR is a bonus; never fail because of it
        log.debug("OCR skipped: %s", e)
        return ""


# --------------------------------------------------------------------------- cleaning
def _clean_line(line: str) -> str:
    line = line.replace(" ", " ").replace("﻿", "").replace("\x02", "-").rstrip()
    line = line.replace("\t", " ")                       # some PDF libraries separate words with tabs
    line = _ZERO_WIDTH.sub("", line)                      # zero-width / soft-hyphen characters split tokens
    line = re.sub(r" {2,}", " ", line)
    line = BULLET_RE.sub("- ", line.strip())
    return line.strip()


def parse_manual(raw: str) -> list[Doc]:
    """Split raw text into documents and clean them."""
    lines = raw.replace("\r", "").replace("\f", "\n").split("\n")
    docs_pages: list[list[list[str]]] = []
    doc_pages: list[list[str]] = []
    page: list[str] = []
    for ln in lines:
        m = FOOTER_RE.match(ln)
        if m:
            doc_pages.append(page)
            page = []
            if int(m.group(1)) >= int(m.group(2)):
                docs_pages.append(doc_pages)
                doc_pages = []
            continue
        if FILE_RE.match(ln):
            continue
        page.append(ln)
    if page:
        doc_pages.append(page)
    if doc_pages:
        docs_pages.append(doc_pages)

    docs: list[Doc] = []
    for pages in docs_pages:
        doc = _build_doc(len(docs), pages)
        if doc.lines:
            docs.append(doc)
    _inherit_sections(docs)
    return docs


def _build_doc(index: int, pages: list[list[str]]) -> Doc:
    doc = Doc(index=index)
    for pg in pages:
        kept: list[str] = []
        i = 0
        while i < len(pg):
            ln = _clean_line(pg[i])
            i += 1
            if not ln:
                continue
            hm = HEADER_RE.match(ln)
            if hm:
                sid, rest = hm.group(1), hm.group(2)
                vm = VEHICLE_TAIL_RE.search(rest)
                if vm:
                    doc.vehicle = vm.group(1)
                    rest = rest[:vm.start()]
                if not doc.section_id:
                    doc.section_id, doc.section_title = sid, rest.strip()
                # the mode line follows (possibly after blank lines)
                j = i
                while j < len(pg) and not pg[j].strip():
                    j += 1
                if j < len(pg):
                    mm = MODE_RE.match(_clean_line(pg[j]))
                    if mm:
                        doc.mode = doc.mode or mm.group(1).strip()
                        i = j + 1
                continue
            if MODE_RE.match(ln) or STRAY_RE.match(ln):
                mm = MODE_RE.match(ln)
                if mm:
                    doc.mode = doc.mode or mm.group(1).strip()
                continue
            kept.append(ln)
        doc.pages.append("\n".join(kept))
        doc.lines.extend(kept)
    doc.lines = reflow(doc.lines)          # rebuild tables that the PDF library split into one cell per line
    doc.title = _doc_title(doc)
    return doc


def _inherit_sections(docs: list[Doc]) -> None:
    """A document without a header (rare) belongs to the section of the previous one."""
    last = None
    for d in docs:
        if d.section_id:
            last = d
        elif last is not None:
            d.section_id, d.section_title = last.section_id, last.section_title
            d.vehicle = d.vehicle or last.vehicle


def _doc_title(doc: Doc) -> str:
    for ln in doc.lines[:4]:
        if 2 <= len(ln) <= 80 and not ln.endswith((".", ":")) and not ln[0].isdigit() and not ln.startswith("-"):
            if ln.lower() in {"material", "torque specifications", "specifications"}:
                continue
            return ln
    return doc.section_title


def load_documents(path: str | Path) -> list[Doc]:
    """Read + clean a manual. Returns the list of documents."""
    return load_documents_info(path)[0]


def load_documents_info(path: str | Path) -> tuple[list[Doc], dict, str]:
    """Like ``load_documents`` but also returns the extraction info and the raw text (for diagnostics)."""
    raw, info = read_raw_info(path)
    return parse_manual(raw), info, raw
