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

log = logging.getLogger(__name__)

FOOTER_RE = re.compile(r"^\s*.*?\bPage\s+(\d+)\s+(?:sur|of)\s+(\d+)\s*$", re.I)
FILE_RE = re.compile(r"^\s*file:///", re.I)
HEADER_RE = re.compile(r"^SECTION\s+(\d{3}-\d{2}[A-Z]?)\s*:\s*(.+?)\s*$")
VEHICLE_TAIL_RE = re.compile(r"\s+(\d{4}\s+\S.*?Workshop Manual)\s*$", re.I)
MODE_RE = re.compile(r"^([A-Z][A-Z &/,\-]+?)\s+Procedure revision date:.*$")
BULLET_RE = re.compile(r"^[z•●\x84]\s+")
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
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manual not found: {p}")
    if p.suffix.lower() in {".txt", ".text", ".md"}:
        return p.read_text(encoding="utf-8", errors="replace")
    return _read_pdf(p)


def _read_pdf(p: Path) -> str:
    errors = []
    try:  # best quality, and gives us the OCR hook
        import fitz  # PyMuPDF
        out = []
        with fitz.open(p) as pdf:
            for page in pdf:
                text = page.get_text("text")
                if len(text.strip()) < 20:
                    text = _ocr_page(page) or text
                out.append(text)
        return "\n".join(out)
    except ImportError as e:
        errors.append(f"pymupdf: {e}")
    try:
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
    except ImportError as e:
        errors.append(f"pypdf: {e}")
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(p))
    except ImportError as e:
        errors.append(f"pdfminer: {e}")
    raise RuntimeError("No PDF library available (" + "; ".join(errors) + "). pip install pymupdf")


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
    line = re.sub(r"[ \t]{2,}", " ", line)
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
    return parse_manual(read_raw(path))
