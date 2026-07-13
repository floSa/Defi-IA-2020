"""Gradient boosting — cheval de bataille (CPU).

LightGBM (défaut) ou CatBoost, entraînés avec un objectif adapté à la MAE :
``mae`` (L1), ``huber`` ou ``quantile`` (alpha=0.5). Produit des prédictions **out-of-fold**
(GroupKFold par thread) pour le blending, plus les prédictions test.

Option deux étages (cf. docs/plan.md §3) : classifieur downvoté/normal/viral puis régression
par régime — souvent plus robuste sur la queue lourde.
TODO (post-validation) : implémenter l'entraînement OOF + prédiction test.
"""
from __future__ import annotations


def train_oof(X, y, groups, cfg):
    """Entraîne le GBM en GroupKFold, retourne (oof_pred, test_pred, models, mae)."""
    raise NotImplementedError("À implémenter après validation du plan.")
