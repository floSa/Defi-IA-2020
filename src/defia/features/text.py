"""Text mining — représentations lexicales et sémantiques (volet 30 pts du barème).

  * TF-IDF word + char n-grams (hashing pour tenir en RAM), optionnellement réduit par SVD
    pour injection dans le GBM. Sert aussi de vue « texte pur » à un modèle linéaire.
  * Embeddings de phrase modernes (e5 / bge / gte / ModernBERT) — étape C, GPU. Injectés dans
    le GBM (late fusion tabulaire + texte).
  * Topics : clustering d'embeddings / BERTopic ; distance topique commentaire<->thread.

Voir docs/compute_strategy.md pour l'exécution des parties GPU (desktop / Kaggle / Colab).
TODO (post-validation) : implémenter TF-IDF (CPU) puis embeddings (GPU).
"""
from __future__ import annotations


def build_tfidf(train_texts, test_texts, **cfg):
    raise NotImplementedError("À implémenter après validation du plan.")


def build_sentence_embeddings(texts, model_name: str, batch_size: int = 256):
    """Étape C (GPU). Extrait des embeddings de phrase pour la late fusion."""
    raise NotImplementedError("À implémenter après validation du plan.")
