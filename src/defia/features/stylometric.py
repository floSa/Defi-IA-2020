"""Text mining — features stylométriques et de sentiment (volet 30 pts du barème).

Familles (cf. docs/plan.md §2b) :
  * Longueurs : caractères, mots, phrases, longueur moyenne des mots.
  * Casse & ponctuation : ratio majuscules, comptages de ``!``, ``?``, ``!!``, ``...``.
  * Marqueurs : présence d'URL, de markdown (``**``, citations ``>``), d'émoticônes/emoji.
  * Lexique : richesse (type-token ratio), profanité, lisibilité (Flesch).
  * Sentiment : VADER (compound, pos/neg/neu) — rapide et CPU-friendly.

TODO (post-validation) : implémenter les extracteurs.
"""
from __future__ import annotations


def add_length_features(df, text_col: str = "body"):
    raise NotImplementedError("À implémenter après validation du plan.")


def add_punctuation_features(df, text_col: str = "body"):
    raise NotImplementedError("À implémenter après validation du plan.")


def add_sentiment_features(df, text_col: str = "body"):
    raise NotImplementedError("À implémenter après validation du plan.")
