"""Baselines MAE.

  * B0 : constante = médiane de ``ups`` (= 1 sur ce corpus). Plancher de référence.
  * B1 : médiane par groupe (thread / heure de la journée / auteur), avec repli sur la
    médiane globale pour les groupes absents du train.

La MAE étant minimisée par la médiane, ces baselines sont volontairement forts : ils fixent
la barre que les modèles appris doivent battre.
TODO (post-validation) : implémenter B0/B1.
"""
from __future__ import annotations


def predict_global_median(train_y, n_test: int):
    raise NotImplementedError("À implémenter après validation du plan.")


def predict_group_median(train_df, test_df, group_col: str, target: str = "ups"):
    raise NotImplementedError("À implémenter après validation du plan.")
