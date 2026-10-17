"""Streamlit UI (bonus):  streamlit run app.py"""
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


@st.cache_resource(show_spinner="Indexing the manual...")
def load(path, extract_backend):
    return Pipeline(path, extract_backend=extract_backend)


with st.sidebar:
    upload = st.file_uploader("Service manual (.pdf / .txt)", type=["pdf", "txt"])
    st.caption("Extractor: rule-based (offline; no API key required)")
    backend = "rules"
    top_k = st.slider("Chunks retrieved", 3, 15, config.TOP_K)

src = None
if upload:
    config.DATA_DIR.mkdir(exist_ok=True)
    src = config.CACHE_DIR / upload.name
    config.CACHE_DIR.mkdir(exist_ok=True)
    src.write_bytes(upload.getvalue())
pipe = load(str(src) if src else None, backend)
st.caption(f"{pipe.source.name}: {len(pipe.pages)} pages, {len(pipe.chunks)} chunks, "
           f"embeddings={pipe.retriever.embedder.backend}, extractor={pipe.extractor.name}")

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
