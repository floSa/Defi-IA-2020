"""Text mining — représentation lexicale (TF-IDF) et sémantique (volet 30 pts du barème).

  * TF-IDF : ``HashingVectorizer`` (stateless, mémoire-léger, pas de vocabulaire à stocker sur
    4,2 M documents) word n-grams 1-2, réduit par SVD (``TruncatedSVD``, ajustée sur un
    échantillon) pour injection dense dans le GBM. Aucune fuite possible (pas de label utilisé).
  * Embeddings de phrase modernes (e5/bge/gte/ModernBERT) — étape C, GPU, cf.
    docs/compute_strategy.md.
"""
from __future__ import annotations

import numpy as np


def fit_svd_on_sample(texts_sample, n_features: int, ngram_range: tuple[int, int], n_components: int, seed: int):
    """Ajuste HashingVectorizer + TruncatedSVD sur un échantillon (mémoire-léger)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(
        n_features=n_features, ngram_range=ngram_range,
        alternate_sign=False, norm="l2", lowercase=True,
    )
    X = hv.transform(texts_sample)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(X)
    return hv, svd


def transform_tfidf_svd(texts, hv, svd) -> np.ndarray:
    """Applique le pipeline HashingVectorizer -> SVD à un batch de textes. Retourne un array dense."""
    X = hv.transform(texts)
    return svd.transform(X).astype(np.float32)
