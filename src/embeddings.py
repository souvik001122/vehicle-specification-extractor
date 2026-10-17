"""Step 3 - embeddings and vector index.

Backends (``EMBED_BACKEND``):

* ``sbert``  - sentence-transformers (default model all-MiniLM-L6-v2), runs locally
* ``openai`` - OpenAI embeddings API (needs OPENAI_API_KEY)
* ``tfidf``  - TF-IDF + truncated SVD (LSA). No downloads, no API key, deterministic.
* ``auto``   - sbert if it is installed and loads, otherwise tfidf

The vector index is FAISS (inner product on L2-normalised vectors = cosine) when installed,
otherwise a plain numpy matrix product.  Matrices from sbert/openai are cached in ``.cache/``.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

import config
from .textutil import tokenize

log = logging.getLogger(__name__)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype("float32")


class Embedder:
    """Turns texts into unit vectors. ``fit_transform`` indexes the corpus, ``encode_query`` embeds a question."""

    def __init__(self, backend: str | None = None, cache_dir: Path | None = None):
        self.requested = (backend or config.EMBED_BACKEND).lower()
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self.backend = ""
        self._model = None
        self._client = None
        self._tfidf = None
        self._svd = None
        self._resolve()

    # ------------------------------------------------------------------ setup
    def _resolve(self) -> None:
        req = self.requested
        if req in ("sbert", "auto"):
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(config.SBERT_MODEL)
                self.backend = "sbert"
                return
            except Exception as e:  # not installed, or model cannot be downloaded
                if req == "sbert":
                    raise RuntimeError(f"sentence-transformers backend unavailable: {e}") from e
                log.info("sentence-transformers unavailable (%s) - using TF-IDF/LSA", e)
        if req == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("EMBED_BACKEND=openai needs OPENAI_API_KEY")
            from openai import OpenAI
            self._client = OpenAI()
            self.backend = "openai"
            return
        if req not in ("auto", "tfidf", "sbert"):
            raise ValueError(f"Unknown embedding backend: {req}")
        self.backend = "tfidf"

    # ------------------------------------------------------------------ corpus
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self.backend == "tfidf":
            return self._fit_tfidf(texts)
        key = hashlib.sha1((self.backend + config.SBERT_MODEL + config.OPENAI_EMBED_MODEL + "\x00".join(texts)).encode()).hexdigest()[:20]
        path = self.cache_dir / f"emb-{self.backend}-{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = _normalize(self._encode(texts))
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, vecs)
        except OSError:
            pass
        return vecs

    def encode_query(self, text: str) -> np.ndarray:
        if self.backend == "tfidf":
            v = self._svd.transform(self._tfidf.transform([text]))
            return _normalize(v)
        return _normalize(self._encode([text]))

    # ------------------------------------------------------------------ backends
    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sbert":
            return np.asarray(self._model.encode(texts, batch_size=64, show_progress_bar=False))
        if self.backend == "openai":
            out = []
            for i in range(0, len(texts), 96):
                batch = [t[:8000] for t in texts[i:i + 96]]
                resp = self._client.embeddings.create(model=config.OPENAI_EMBED_MODEL, input=batch)
                out.extend(d.embedding for d in resp.data)
            return np.asarray(out)
        raise RuntimeError("no dense backend")

    def _fit_tfidf(self, texts: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(analyzer=lambda t: _ngrams(tokenize(t)), sublinear_tf=True, min_df=1)
        x = self._tfidf.fit_transform(texts)
        dims = max(2, min(config.LSA_DIMS, x.shape[0] - 1, x.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=dims, random_state=0)
        return _normalize(self._svd.fit_transform(x))


def _ngrams(tokens: list[str]) -> list[str]:
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


class VectorIndex:
    """Cosine nearest-neighbour search: FAISS if available, otherwise numpy."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self._faiss = None
        try:
            import faiss
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
        except Exception:
            self._faiss = None

    @property
    def kind(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self.vectors))
        q = np.ascontiguousarray(query_vec, dtype="float32").reshape(1, -1)
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        sims = self.vectors @ q[0]
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]
