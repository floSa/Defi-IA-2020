"""Métriques d'évaluation. La métrique officielle du challenge est la MAE (à minimiser).

Rappel : la MAE est minimisée par la **médiane** conditionnelle. Toutes les prédictions sont
évaluées dans l'espace original de la cible, même si le modèle a été entraîné sur ``log1p``.
"""
from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error dans l'espace original de la cible."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mae_report(y_true, y_pred, baseline_value: float = 1.0) -> dict[str, float]:
    """MAE du modèle vs baseline constant (médiane), avec le gain relatif.

    ``baseline_value`` par défaut = 1.0 (médiane empirique de ``ups`` sur le train).
    """
    y_true = np.asarray(y_true, dtype=float)
    model = mae(y_true, y_pred)
    base = mae(y_true, np.full_like(y_true, baseline_value))
    return {
        "mae": model,
        "mae_baseline": base,
        "gain_abs": base - model,
        "gain_rel": (base - model) / base if base else 0.0,
    }
