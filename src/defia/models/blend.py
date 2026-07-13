"""Blending des prédictions out-of-fold.

Combine les OOF de M1 (GBM), M3 (linéaire TF-IDF), M4 (transformer), M5 (late fusion) via un
méta-modèle optimisant la MAE : ridge, LightGBM, ou NNLS (poids positifs). Les poids sont
appris sur les OOF puis appliqués aux prédictions test. Résultat -> ``submissions/``.
TODO (post-validation) : implémenter le meta-learner.
"""
from __future__ import annotations


def blend_oof(oof_matrix, y, test_matrix, method: str = "ridge"):
    raise NotImplementedError("À implémenter après validation du plan.")
