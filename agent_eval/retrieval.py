"""Deterministic in-memory BM25 over the frozen corpus.

Implemented directly on a scipy sparse matrix (no external retriever) so that the
index is bit-for-bit reproducible across Kaggle sessions and library versions.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his in is it its of on or she
    that the their them they this to was were which who whom whose with what when where why
    how did does do i you we our your than then there these those been being also into""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.doc_ids: list[str] = []
        self.matrix: sparse.csc_matrix | None = None  # terms x docs, BM25 weights
        self.idf: np.ndarray | None = None

    def fit(self, docs: list[dict[str, Any]]) -> "BM25Index":
        rows, cols, vals = [], [], []
        doc_lens = np.zeros(len(docs), dtype=np.float64)
        for j, d in enumerate(docs):
            self.doc_ids.append(d["doc_id"])
            toks = tokenize(f"{d['title']} {d['text']}")
            doc_lens[j] = len(toks)
            counts: dict[int, int] = {}
            for t in toks:
                idx = self.vocab.setdefault(t, len(self.vocab))
                counts[idx] = counts.get(idx, 0) + 1
            for idx, c in counts.items():
                rows.append(idx)
                cols.append(j)
                vals.append(c)
        n_docs = len(docs)
        tf = sparse.csc_matrix(
            (np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(len(self.vocab), n_docs)
        )
        df = np.diff(tf.tocsr().indptr).astype(np.float64)
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        avgdl = doc_lens.mean() if n_docs else 1.0
        tf = tf.tocoo()
        norm = self.k1 * (1 - self.b + self.b * doc_lens[tf.col] / avgdl)
        w = self.idf[tf.row] * tf.data * (self.k1 + 1) / (tf.data + norm)
        self.matrix = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        assert self.matrix is not None, "index not fitted"
        term_ids = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not term_ids:
            return []
        scores = np.asarray(self.matrix[term_ids].sum(axis=0)).ravel()
        k = min(k, int((scores > 0).sum()))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        # Ties broken by doc_id order so results are deterministic.
        top = sorted(top, key=lambda j: (-scores[j], j))
        return [(self.doc_ids[j], float(scores[j])) for j in top]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"k1": self.k1, "b": self.b, "vocab": self.vocab, "doc_ids": self.doc_ids,
                 "matrix": self.matrix, "idf": self.idf},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            state = pickle.load(f)
        idx = cls(state["k1"], state["b"])
        idx.vocab, idx.doc_ids = state["vocab"], state["doc_ids"]
        idx.matrix, idx.idf = state["matrix"], state["idf"]
        return idx


def load_or_build_index(data_dir: str | Path, docs: list[dict[str, Any]] | None = None) -> BM25Index:
    data_dir = Path(data_dir)
    path = data_dir / "bm25.pkl"
    if path.exists():
        return BM25Index.load(path)
    if docs is None:
        from agent_eval.data import load_corpus

        docs = load_corpus(data_dir)
    idx = BM25Index().fit(docs)
    try:
        idx.save(path)
    except OSError:
        # Kaggle input datasets are read-only; the in-memory index is still usable.
        pass
    return idx
