"""Blending des prédictions out-of-fold (Milestone E).

Combine les prédictions OOF de plusieurs modèles (variantes GBM, transformer…) sur le **holdout
temporel** (où l'on connaît la vérité `ups`), en cherchant les poids qui minimisent directement
la **MAE** (pas la MSE — cohérent avec la métrique). Les mêmes poids sont ensuite appliqués aux
prédictions test pour produire la soumission finale.

Convention de fichiers (dans ``data/processed/oof/``) : pour chaque modèle ``<name>``, une paire
``oof_<name>.parquet`` (id, pred sur le holdout) et ``test_<name>.parquet`` (id, pred sur le test).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def optimize_weights_mae(oof_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Poids d'une combinaison convexe minimisant la MAE (somme=1, poids >= 0)."""
    n_models = oof_matrix.shape[1]

    def mae_loss(w):
        return float(np.mean(np.abs(y - oof_matrix @ w)))

    w0 = np.full(n_models, 1.0 / n_models)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * n_models
    res = minimize(mae_loss, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-8})
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else w0
