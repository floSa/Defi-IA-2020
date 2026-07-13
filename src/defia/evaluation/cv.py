"""Schéma de validation croisée.

On utilise **GroupKFold par ``link_id`` (thread)** : les commentaires d'un même thread
partagent des features (timing, taille du thread, arbre de réponses) et des cibles corrélées.
Grouper par thread évite la fuite et donne une estimation MAE honnête, cohérente avec le fait
que le test contient des threads entiers.

Toute feature à risque de fuite (target encoding auteur/thread) doit être calculée **à
l'intérieur du fold d'entraînement** puis appliquée au fold de validation.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


def group_kfold_indices(
    groups, n_splits: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Rend les (train_idx, valid_idx) d'un GroupKFold mélangé et déterministe.

    Implémenté par assignation de chaque groupe unique à un fold (shuffle graine-déterministe),
    ce qui évite qu'un même thread se retrouve à cheval sur train et validation.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_of_group = {g: i % n_splits for i, g in enumerate(unique)}
    fold = np.array([fold_of_group[g] for g in groups])
    for k in range(n_splits):
        valid_idx = np.where(fold == k)[0]
        train_idx = np.where(fold != k)[0]
        yield train_idx, valid_idx
