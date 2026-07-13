"""Schémas de validation.

Le split officiel train/test est **temporel** (train = 1–24 mai, test = 25–31 mai 2015 ;
cf. docs/eda_findings.md). La validation de référence est donc un **holdout temporel** : on
apprend sur le début du train et on valide sur les N derniers jours, pour reproduire l'horizon
du test (7 jours). C'est ``temporal_holdout_indices`` — **le juge de paix**.

``group_kfold_indices`` (GroupKFold par ``link_id``) reste utile pour des diagnostics et pour
entraîner des modèles robustes sans fuite intra-thread, mais il **surestime** la performance sur
ce problème temporel : ne pas s'y fier comme métrique finale.

Toute feature à risque de fuite (target encoding auteur/thread) doit être calculée **sur le passé
uniquement** (le holdout temporel le garantit) ou dans le fold d'entraînement.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

# 7 derniers jours de train = même horizon que le test (25–31 mai).
SECONDS_PER_DAY = 86_400


def temporal_holdout_indices(
    created_utc, val_days: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """(fit_idx, val_idx) : validation = les ``val_days`` derniers jours (par ``created_utc``).

    Reproduit l'horizon du test officiel. Le fit ne voit que des commentaires antérieurs au
    début de la fenêtre de validation → pas de fuite temporelle.
    """
    t = np.asarray(created_utc, dtype=np.int64)
    cutoff = int(t.max()) - val_days * SECONDS_PER_DAY
    val_idx = np.where(t > cutoff)[0]
    fit_idx = np.where(t <= cutoff)[0]
    return fit_idx, val_idx


def time_series_folds(
    created_utc, n_splits: int = 4, val_days: int = 7
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """CV temporelle expansive : fenêtres de validation glissantes de ``val_days`` jours,
    chaque fit n'utilisant que le passé. Utile pour une estimation MAE plus stable que le
    holdout unique.
    """
    t = np.asarray(created_utc, dtype=np.int64)
    t_max = int(t.max())
    for k in range(n_splits, 0, -1):
        val_hi = t_max - (k - 1) * val_days * SECONDS_PER_DAY
        val_lo = val_hi - val_days * SECONDS_PER_DAY
        val_idx = np.where((t > val_lo) & (t <= val_hi))[0]
        fit_idx = np.where(t <= val_lo)[0]
        if len(fit_idx) and len(val_idx):
            yield fit_idx, val_idx


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
