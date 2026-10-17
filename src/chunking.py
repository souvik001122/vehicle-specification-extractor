"""Step 2 - structure-aware chunking.

Fixed-size windows would cut a torque table in half and lose whether a row belongs to the
front or the rear brake.  Instead we use the manual's own structure:

* every chunk knows its section (``206-03: Front Disc Brake``), mode and heading;
* specification tables (torque / measurements / materials) and parts tables are kept whole
  and become their own chunk kind;
* all remaining text is packed into ~``MAX_CHUNK_CHARS`` pieces, split at headings, with a
  small line overlap so a sentence at the boundary is seen twice.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import config
from .pdf_extract import Doc
from .tables import consume_parts, table_start

TABLE_KINDS = {"torque_table", "spec_table", "material_table", "parts_table"}
_TABLE_TITLE = {
    "torque_table": "Torque Specifications",
    "spec_table": "Specifications",
    "material_table": "Materials, fluids and fill capacity",
    "parts_table": "Parts list (Item, Part Number, Description)",
}
PHASES = {
    "removal", "installation", "disassembly", "assembly", "adjustment", "inspection", "removal and installation",
    "initial disassembly", "final assembly", "draining and filling", "bleeding", "cleaning", "all vehicles",
    "disassembly of subassemblies", "assembly of subassemblies",
}
_SMALL = {"and", "or", "of", "the", "in", "to", "for", "with", "a", "an", "on", "—", "-", "/", "at", "by", "from"}


@dataclass
class Chunk:
    id: int
    doc_index: int
    section_id: str
    section_title: str
    mode: str
    heading: str
    kind: str                       # text | torque_table | spec_table | material_table | parts_table
    text: str

    @property
    def section(self) -> str:
        return f"{self.section_id}: {self.section_title}" if self.section_id else self.section_title

    @property
    def embed_text(self) -> str:
        """Text that is embedded / indexed: metadata prefix + body."""
        return f"Section {self.section} | {self.mode} | {self.heading}\n{self.text}"

    @property
    def is_table(self) -> bool:
        return self.kind in TABLE_KINDS


def is_heading(line: str) -> bool:
    """Short Title-Case line without sentence punctuation, e.g. 'Brake Caliper Anchor Plate'."""
    if len(line) < 3 or len(line) > 70 or line.endswith((".", ":", ",", ";")) or line[0] in "-0123456789(":
        return False
    words = line.split()
    if not 1 <= len(words) <= 9:
        return False
    ok = all(w[0].isupper() or w.lower() in _SMALL or not w[0].isalpha() for w in words)
    return ok and any(w[0].isupper() for w in words)


def _split_blocks(doc: Doc) -> list[tuple[str, list[str]]]:
    """Split a document into (kind, lines) blocks."""
    lines = doc.lines
    n = len(lines)
    blocks: list[tuple[str, list[str]]] = []
    text: list[str] = []

    def flush():
        if text:
            blocks.append(("text", text[:]))
            text.clear()

    i = 0
    while i < n:
        ts = table_start(lines, i)
        if ts is None:
            text.append(lines[i])
            i += 1
            continue
        kind, hl = ts
        if kind == "parts_table":
            end = consume_parts(lines, i + hl)
            flush()
            blocks.append((kind, lines[i:end]))
            i = end
            continue
        if not doc.is_spec:          # material / spec tables outside spec pages stay inside the text
            text.append(lines[i])
            i += 1
            continue
        end = i + hl
        while end < n and table_start(lines, end) is None:
            end += 1
        flush()
        blocks.append((kind, lines[i:end]))
        i = end
    flush()
    return blocks


def _pack_text(lines: list[str], max_chars: int, overlap: int) -> list[tuple[str, list[str]]]:
    """Pack text lines into chunks; returns (heading_at_start, lines)."""
    out: list[tuple[str, list[str]]] = []
    cur: list[str] = []
    fresh = 0                       # lines in ``cur`` that are not overlap from the previous chunk
    size = 0
    heading = cur_heading = ""

    def cut():
        nonlocal cur, fresh, size, cur_heading
        out.append((cur_heading, cur))
        cur = cur[-overlap:] if overlap else []
        fresh, size, cur_heading = 0, sum(len(x) + 1 for x in cur), heading

    for ln in lines:
        head_line = is_heading(ln) or ln.lower() in PHASES
        if head_line and fresh and size > max_chars * 0.5:
            cut()
        if not cur:
            cur_heading = heading
        cur.append(ln)
        fresh += 1
        size += len(ln) + 1
        if head_line and ln.lower() not in PHASES:
            heading = ln
        elif head_line:
            heading = (heading.split(" — ")[0] + " — " + ln.title()) if heading else ln.title()
        if size >= max_chars:
            cut()
    if fresh:
        out.append((cur_heading, cur))
    return out


def _split_table(kind: str, lines: list[str], max_chars: int) -> list[list[str]]:
    """Very long tables are split by rows, repeating the header line(s)."""
    if sum(len(x) + 1 for x in lines) <= max_chars * 2:
        return [lines]
    header = lines[:1]
    parts, cur, size = [], [], 0
    for ln in lines[1:]:
        cur.append(ln)
        size += len(ln) + 1
        if size >= max_chars * 1.5:
            parts.append(header + cur)
            cur, size = [], 0
    if cur:
        parts.append(header + cur)
    return parts


def chunk_documents(docs: list[Doc], max_chars: int | None = None, overlap: int | None = None) -> list[Chunk]:
    max_chars = max_chars or config.MAX_CHUNK_CHARS
    overlap = config.CHUNK_OVERLAP_LINES if overlap is None else overlap
    chunks: list[Chunk] = []

    def add(doc: Doc, kind: str, heading: str, body: list[str]):
        text = "\n".join(body).strip()
        if len(text) < 12:
            return
        chunks.append(Chunk(len(chunks), doc.index, doc.section_id, doc.section_title, doc.mode, heading, kind, text))

    for doc in docs:
        for kind, lines in _split_blocks(doc):
            if kind == "text":
                for heading, body in _pack_text(lines, max_chars, overlap):
                    add(doc, "text", heading or doc.title, body)
            else:
                title = f"{doc.title} — {_TABLE_TITLE[kind]}" if not doc.is_spec else _TABLE_TITLE[kind]
                for part in _split_table(kind, lines, max_chars):
                    body = [_TABLE_TITLE[kind]] + part if kind == "torque_table" else part
                    add(doc, kind, title, body)
    return chunks
