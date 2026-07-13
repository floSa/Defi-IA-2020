"""Text mining — features stylométriques et de sentiment (volet 30 pts du barème).

Toutes calculées sur le `body` (déséchappé des entités HTML). Vectorisées via l'accessor
`.str` de pandas pour tenir la volumétrie (4,2 M lignes). Le sentiment VADER est optionnel
(plus lent, pur Python).

Familles (cf. docs/plan.md §2b) : longueurs, casse, ponctuation, marqueurs (URL, markdown,
émoticônes), richesse lexicale, sentiment.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from defia.data.load import unescape_body

_URL_RE = r"https?://\S+"
_EMOTICON_RE = r"[:;=8]['`\-]?[)(\]\[dDpP/\\|]"  # :) ;) :-D :( :/ ... approximation
_DELETED = {"[deleted]", "[removed]", ""}


def build_stylometric_features(df: pd.DataFrame, with_sentiment: bool = True) -> pd.DataFrame:
    """Features stylométriques pour un split. Attend colonnes : id, body."""
    out = pd.DataFrame(index=df.index)
    out["id"] = df["id"].to_numpy()

    body = unescape_body(df["body"])  # entités HTML décodées, NA -> ""
    s = body.str

    n_chars = s.len().fillna(0).to_numpy()
    words = body.str.split()
    n_words = words.str.len().fillna(0).to_numpy()
    n_unique = words.map(lambda w: len(set(w)) if w else 0).to_numpy()

    out["n_chars"] = n_chars.astype(np.int32)
    out["n_words"] = n_words.astype(np.int32)
    out["avg_word_len"] = (n_chars / np.maximum(n_words, 1)).astype(np.float32)
    out["unique_word_ratio"] = (n_unique / np.maximum(n_words, 1)).astype(np.float32)
    out["n_sentences"] = s.count(r"[.!?]+").fillna(0).to_numpy().astype(np.int32)

    n_upper = s.count(r"[A-Z]").fillna(0).to_numpy()
    n_letters = s.count(r"[A-Za-z]").fillna(0).to_numpy()
    n_digits = s.count(r"[0-9]").fillna(0).to_numpy()
    out["uppercase_ratio"] = (n_upper / np.maximum(n_letters, 1)).astype(np.float32)
    out["digit_ratio"] = (n_digits / np.maximum(n_chars, 1)).astype(np.float32)

    out["n_exclaim"] = s.count("!").fillna(0).to_numpy().astype(np.int32)
    out["n_question"] = s.count(r"\?").fillna(0).to_numpy().astype(np.int32)
    out["has_multi_exclaim"] = s.contains(r"!!").fillna(False).to_numpy()
    out["has_ellipsis"] = s.contains(r"\.\.\.").fillna(False).to_numpy()
    out["ends_question"] = body.str.rstrip().str.endswith("?").fillna(False).to_numpy()

    out["n_urls"] = s.count(_URL_RE).fillna(0).to_numpy().astype(np.int32)
    out["has_url"] = (out["n_urls"] > 0)
    out["has_bold"] = s.contains(r"\*\*").fillna(False).to_numpy()
    out["has_quote"] = s.contains(r"(?:^|\n)\s*>").fillna(False).to_numpy()
    out["n_newlines"] = s.count("\n").fillna(0).to_numpy().astype(np.int32)
    out["n_emoticons"] = s.count(_EMOTICON_RE).fillna(0).to_numpy().astype(np.int32)

    out["is_deleted_body"] = body.isin(_DELETED).to_numpy()

    if with_sentiment:
        out = out.join(_vader_features(body))

    return out


_SIA = None


def _get_sia():
    """Analyseur VADER mis en cache au niveau module (évite de recharger le lexique par chunk)."""
    global _SIA
    if _SIA is None:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        _SIA = SentimentIntensityAnalyzer()
    return _SIA


def _vader_features(body: pd.Series) -> pd.DataFrame:
    """Scores VADER (compound, pos, neg, neu). Pur Python -> plus lent."""
    polarity = _get_sia().polarity_scores
    # cache sur textes vides / très courts pour accélérer
    vals = np.empty((len(body), 4), dtype=np.float32)
    for i, txt in enumerate(body.to_numpy()):
        d = polarity(txt) if txt else {"compound": 0, "pos": 0, "neg": 0, "neu": 0}
        vals[i] = (d["compound"], d["pos"], d["neg"], d["neu"])
    return pd.DataFrame(
        vals, index=body.index,
        columns=["vader_compound", "vader_pos", "vader_neg", "vader_neu"],
    )
