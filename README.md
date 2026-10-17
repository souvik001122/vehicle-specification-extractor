# Vehicle Specification Extraction (RAG)

Ask a question such as *"Torque for brake caliper bolts"* and get structured specs from a
vehicle service manual (2014 F-150 Workshop Manual sample):

```json
[{"component": "Brake caliper guide pin bolts", "spec_type": "Torque", "value": "37", "unit": "Nm", "section": "206-03: Front Disc Brake"}]
```

`section` is an extra field beyond the four requested (`--minimal` removes it). It matters because front and
rear brakes share component names but have different values (guide pin bolts: 37 Nm front, 33 Nm rear).

## Quick start

```bash
pip install -r requirements.txt          # minimum: numpy scikit-learn pypdf
python -m src.main --query "Torque for brake caliper bolts"
python -m src.main --queries-file queries.txt --out outputs/results.csv     # or .json
streamlit run app.py                     # optional UI
python tests/test_pipeline.py            # or: pytest -q
```

The manual is in `data/` (`sample-service-manual.txt`, a text dump of the PDF). A real `.pdf` works too:
drop it in `data/` or pass `--source path/to/manual.pdf`.

This submission uses **offline rule-based extraction** and requires no API key or paid API. Retrieval uses hybrid ranking (dense embeddings + BM25/RRF) and then prioritizes chunks that contain all requested component keywords when exact matches exist.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1. Extract + clean | `src/pdf_extract.py` | PyMuPDF, falling back to pypdf then pdfminer; optional OCR for scanned pages. Removes footers (`Page X sur Y`), `file:///` lines and revision dates; converts the Wingdings bullet `z` to `-`; footers become page breaks. |
| 2. Chunk | `src/chunking.py` | Uses the manual's own structure: `SECTION 206-03: Front Disc Brake` and mode lines (SPECIFICATIONS, REMOVAL AND INSTALLATION...) become metadata. Spec tables are kept whole as their own chunks; other text is packed into ~1400 character chunks with a 2-line overlap. Embedding text is prefixed with section and heading. |
| 3. Embed + index | `src/embeddings.py` | sentence-transformers (auto-detected), OpenAI embeddings, or TF-IDF+LSA fallback. FAISS if installed, numpy cosine otherwise. Cached in `.cache/`. |
| 4. Retrieve | `src/retriever.py` | Hybrid: dense vectors + BM25, merged with reciprocal rank fusion, small boost for spec tables on quantitative questions, light query expansion (torque -> tighten, Nm). |
| 5. Extract | `src/extractor.py` | Offline rule parser for torque tables, "Tighten to 37 Nm (27 lb-ft)" steps, measurements (minimum thickness) and fluid capacities. |
| 6. Output | `src/main.py` | JSON to stdout; `--out` writes `.json` or `.csv`. |

### Design decisions

- **Section-aware chunking.** Fixed-size windows would split torque tables and lose which section (front or rear) a value belongs to.
- **Hybrid retrieval.** Part names and units need exact matching; paraphrases need embeddings.
- **Evidence-grounded extraction.** The rule parser reads values from retrieved table rows and procedure text, preserving the component name and section.
- **Torque is reported in Nm.** The lb-ft/lb-in alternatives in the manual are not reported. Rows that give only lb-in use the Nm column when present.
- **Manual quirks handled.** Component names stripped from procedure steps are recovered from nearby fastener phrases or the page heading; `—` placeholders in table columns are ignored; wrapped table row names are rejoined.

## Known limitations / ideas for improvement

- The rule parser only knows Nm/lb-ft/lb-in, mm, ml/L. Staged torques (e.g. 35 then 70 Nm, or "+ 90 degrees") are returned as separate values without stage labels; the angle part is not captured.
- Fluid-capacity tables are wrapped text with no clear structure, so the offline rules can return extra rows (e.g. small additive quantities).
- Add a cross-encoder re-ranker and query rewriting (e.g. "caliper bolts" -> "guide pin / anchor plate / flow bolt").
- Use a vehicle/engine-variant filter (e.g. 4x2 vs 4x4, SVT Raptor) as retrieval metadata.
- Parse tables with layout-aware extraction (PyMuPDF `find_tables`) and keep page images for tables that were scanned.
- Evaluate with a labelled question set (precision/recall on component+value) and add an LLM-as-judge check.
- Persist the index in FAISS/Chroma for much larger manuals.

## Environment variables

`EMBED_BACKEND`, `TOP_K`, `MAX_CHUNK_CHARS`. This submission uses the `rules` extractor by default and does not require an API key.
Defaults are in `config.py`.

## Tests and sample results

- `python tests/test_pipeline.py` (or `pytest -q`) runs 21 offline tests: query parsing, cleaning, chunking, extraction
  checked against values read by hand from the manual (including all six front/rear brake-caliper torque rows, guide pin bolts 37 Nm front / 33 Nm rear, wheel
  nuts 204 Nm, rear brake disc minimum thickness 23.0 mm), and JSON/CSV output.
- `outputs/results.json` and `outputs/results.csv` hold the answers to the 20 questions in `queries.txt`;
  `outputs/example_minimal.json` shows the exact four-field format from the assignment.
- The PDF ingestion path (PyMuPDF -> pypdf -> pdfminer) is supported. A regression test covers a PDF-style flattened torque table represented as a text chunk; validate against your exact PDF because PDF text layout varies by file and extraction engine.
