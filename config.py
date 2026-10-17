"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"
OUTPUT_DIR = ROOT / "outputs"

# ---- chunking ---------------------------------------------------------------
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", 1400))
CHUNK_OVERLAP_LINES = int(os.getenv("CHUNK_OVERLAP_LINES", 2))

# ---- embeddings -------------------------------------------------------------
# auto -> sentence-transformers if installed, else TF-IDF + LSA (no downloads needed)
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "auto")  # auto | sbert | openai | tfidf
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LSA_DIMS = 256

# ---- retrieval --------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", 8))
RRF_K = 60
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 1.2

# ---- LLM extraction ---------------------------------------------------------
# auto -> LLM if an API key is present, otherwise the offline rule-based extractor
EXTRACT_BACKEND = os.getenv("EXTRACT_BACKEND", "rules")  # rules (offline; no API key required)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
