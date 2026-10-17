"""Step 4 - hybrid retrieval.

Dense vectors capture paraphrases ("how tight should the caliper bolts be"), BM25 captures
exact part names and units ("guide pin bolts", "Nm").  The two rankings are merged with
reciprocal rank fusion (RRF).  Spec tables get a small boost because they hold the answers to
quantitative questions, and the query is lightly expanded (torque -> tighten, Nm, lb-ft).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import config
from .chunking import Chunk
from .embeddings import Embedder, VectorIndex
from .textutil import QuerySpec, expand_query, parse_query, synonyms, tokenize

TABLE_BOOST = {"torque_table": 1.35, "material_table": 1.25, "spec_table": 1.25, "parts_table": 1.15}
KIND_FOR_TYPE = {
    "torque": {"torque_table"},
    "capacity": {"material_table"},
    "measurement": {"spec_table", "torque_table"},
    "part_number": {"parts_table", "material_table"},
}


_NM = re.compile(r"\d\s*Nm\b")


class BM25:
    """Okapi BM25 over the stemmed token lists of the chunks."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * len(self.tf)
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.embedder = embedder or Embedder()
        texts = [c.embed_text for c in chunks]
        self.index = VectorIndex(self.embedder.fit_transform(texts))
        self.bm25 = BM25([tokenize(t) for t in texts])

    @staticmethod
    def _coverage(qs: QuerySpec, toks: set[str]) -> float:
        if not qs.keywords:
            return 1.0
        return sum(1 for kw in qs.keywords if synonyms(kw) & toks) / len(qs.keywords)

    def search(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        """Return the ``k`` best (chunk, score) pairs, best first."""
        k = k or config.TOP_K
        qs: QuerySpec = parse_query(query)
        expanded = expand_query(query, qs)
        pool = max(50, k * 6)

        fused: dict[int, float] = {}
        dense = self.index.search(self.embedder.encode_query(expanded), pool)
        for rank, (i, _) in enumerate(dense):
            fused[i] = fused.get(i, 0.0) + config.DENSE_WEIGHT / (config.RRF_K + rank + 1)
        bm = self.bm25.scores(tokenize(expanded))
        for rank, i in enumerate(sorted(range(len(bm)), key=lambda j: -bm[j])[:pool]):
            if bm[i] > 0:
                fused[i] = fused.get(i, 0.0) + config.BM25_WEIGHT / (config.RRF_K + rank + 1)

        preferred = KIND_FOR_TYPE.get(qs.spec_type or "", set())
        for i in list(fused):
            chunk = self.chunks[i]
            toks = set(tokenize(chunk.text + " " + chunk.heading))
            coverage = self._coverage(qs, toks)
            boost = 0.6 + 0.8 * coverage                      # chunks that contain the component words
            if chunk.is_table and coverage == 1.0:            # spec tables hold the answers to quantitative questions
                boost *= TABLE_BOOST.get(chunk.kind, 1.0)
                if chunk.kind in preferred:
                    boost *= 1.15
            elif qs.spec_type == "torque" and coverage == 1.0 and _NM.search(chunk.text):
                boost *= 1.2                                  # a step such as "Tighten to 204 Nm"
            fused[i] *= boost
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])

        # Query-aware precision guard: if the query names a concrete component, prefer
        # chunks that actually mention every component keyword. This prevents a query
        # like "Torque for brake caliper bolts" from filling the result list with generic
        # brake-booster/halfshaft tables merely because they contain the word "brake".
        # Keep the original ranking as a fallback for paraphrased queries where no exact
        # match is present, so recall is not sacrificed unnecessarily.
        if qs.keywords:
            required = set(qs.keywords)
            relevant: list[tuple[int, float]] = []
            for i, score in ranked:
                chunk = self.chunks[i]
                text_tokens = set(tokenize(f"{chunk.heading} {chunk.text}"))
                matched = sum(1 for kw in required if synonyms(kw) & text_tokens)
                coverage = matched / len(required)
                # A named fastener should be present in the passage too, not just in a
                # nearby section. "bolt" also matches the stemmed plural "bolts".
                fastener_ok = not qs.fastener or qs.fastener in text_tokens
                if coverage == 1.0 and fastener_ok:
                    relevant.append((i, score))
            if relevant:
                ranked = relevant

        ranked = ranked[:k]
        return [(self.chunks[i], s) for i, s in ranked]
