"""Baselines MAE.

  * B0 : constante = médiane de ``ups`` (= 1 sur ce corpus). Plancher de référence.
  * B1 : médiane par groupe (heure, auteur, ...), avec repli sur la médiane globale pour les
    groupes absents du fold d'entraînement.

La MAE étant minimisée par la médiane, ces baselines sont volontairement forts : ils fixent la
barre que les modèles appris doivent battre. On les évalue en OOF (GroupKFold par thread) pour
une estimation honnête et comparable aux modèles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_global_median(train_y: pd.Series, n_test: int) -> tuple[np.ndarray, float]:
    """Prédit la médiane du train pour toutes les lignes. Retourne (préds, valeur médiane)."""
    med = float(np.median(np.asarray(train_y, dtype=float)))
    return np.full(n_test, med, dtype=float), med


def predict_group_median(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    group_col: str,
    target: str = "ups",
) -> np.ndarray:
    """Prédit, pour chaque ligne de ``valid_df``, la médiane de ``target`` de son groupe
    (apprise sur ``train_df``), avec repli sur la médiane globale du train.
    """
    global_med = float(train_df[target].median())
    grp_med = train_df.groupby(group_col, observed=True)[target].median()
    return valid_df[group_col].map(grp_med).fillna(global_med).to_numpy(dtype=float)
