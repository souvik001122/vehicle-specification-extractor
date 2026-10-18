"""Streamlit UI (bonus):  streamlit run app.py"""
import hashlib
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from src.extractor import public  # noqa: E402
from src.main import Pipeline  # noqa: E402

st.set_page_config(page_title="Vehicle Spec Extractor", layout="wide")
st.title("Vehicle Specification Extractor")


def code_version() -> str:
    """Short hash of every source file. It is part of the cache key below, so editing any .py file
    (e.g. a new extractor.py) automatically rebuilds the pipeline instead of reusing a stale cached one."""
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for f in sorted([*root.glob("src/*.py"), root / "config.py"]):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:8]


CODE_VERSION = code_version()


@st.cache_resource(show_spinner="Indexing the manual...")
def load(path, extract_backend, content_hash="", version=""):
    # The cache key includes the uploaded file's content hash (a different PDF with the same filename
    # must rebuild the index) and the code version (changed code must not reuse an old cached pipeline).
    return Pipeline(path, extract_backend=extract_backend)


with st.sidebar:
    upload = st.file_uploader("Service manual (.pdf / .txt)", type=["pdf", "txt"])
    st.caption("Extractor: rule-based (offline; no API key required)")
    backend = "rules"
    top_k = st.slider("Chunks retrieved", 3, 15, config.TOP_K)
    st.caption(f"Code version: {CODE_VERSION}")
    if st.button("Clear cache and re-index"):
        st.cache_resource.clear()
        st.rerun()

src = None
content_hash = "default"
if upload:
    payload = upload.getvalue()
    content_hash = hashlib.sha256(payload).hexdigest()
    config.CACHE_DIR.mkdir(exist_ok=True)
    # Unique path prevents same-name uploads from overwriting a file used by a cached pipeline.
    suffix = Path(upload.name).suffix.lower() or ".pdf"
    src = config.CACHE_DIR / f"{Path(upload.name).stem}-{content_hash[:16]}{suffix}"
    if not src.exists():
        src.write_bytes(payload)
pipe = load(str(src) if src else None, backend, content_hash, CODE_VERSION)
st.caption(f"{pipe.source.name}: {len(pipe.pages)} pages, {len(pipe.chunks)} chunks, "
           f"embeddings={pipe.retriever.embedder.backend}, extractor={pipe.extractor.name}, "
           f"text extraction={pipe.extraction.get('strategy', '?')}")
with st.expander("Extraction diagnostics"):
    st.write("Specification rows found by each PDF text-extraction strategy (the best one is used):")
    st.json(pipe.extraction)
    st.download_button("Download extracted text", pipe.raw_text, f"{pipe.source.stem}-extracted.txt")

query = st.text_input("Question", "Torque for brake caliper bolts")
if st.button("Extract", type="primary") and query:
    recs, hits, used = pipe.run(query, top_k)
    rows = [public(r) for r in recs]
    st.subheader(f"Extracted specifications: {len(rows)}")
    st.caption(f"Extractor: {used} · Retrieved evidence chunks: {len(hits)}")
    if rows:
        st.dataframe(rows, use_container_width=True)
        st.download_button("Download JSON", json.dumps(rows, indent=2), "specs.json")
    else:
        st.info("No matching specification found in the retrieved chunks.")
    with st.expander(f"Retrieved evidence ({len(hits)} chunks)"):
        for c, s in hits:
            st.markdown(f"**{c.section_id} {c.section_title}** - {c.heading} (score {s:.4f}, {c.kind})")
            st.code(c.text)
