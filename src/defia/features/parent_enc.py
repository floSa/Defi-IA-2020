"""Network mining avancé — encodage du CONTEXTE PARENT par la cible.

Intuition : répondre à un auteur à succès change la visibilité. On encode :
  * `parent_author_mean` : réputation (moyenne ups lissée, sur le train) de l'AUTEUR du parent.
  * `parent_author_count_log` : activité de l'auteur du parent.

ATTENTION (piège de validation évité) : on N'utilise PAS `parent_ups` (le score du commentaire
parent). Vérification faite : il est connu pour 59 % des lignes du holdout mais seulement 1,5 %
du VRAI test (car dans le test, le parent est presque toujours aussi dans le test, donc `ups`
inconnu). L'inclure gonflait artificiellement la MAE holdout (7,84) sans transférer au
leaderboard. `parent_author_mean`, lui, est un lookup train-global disponible partout : légitime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_parent_encoding(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str = "ups", smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attend train/test avec id, name, author, parent_id, created_utc (+ target sur train)."""
    n_tr = len(train_df)
    union = pd.concat([train_df, test_df], ignore_index=True)
    n = len(union)
    ups_u = np.concatenate([train_df[target].to_numpy(dtype=float), np.full(len(test_df), np.nan)])

    # position du parent
    pos_of_name = pd.Series(np.arange(n), index=union["name"].to_numpy())
    pos_of_name = pos_of_name[~pos_of_name.index.duplicated(keep="first")]
    parent_pos = pos_of_name.reindex(union["parent_id"].to_numpy()).to_numpy()
    del pos_of_name
    has_parent = ~np.isnan(parent_pos)
    pp = np.where(has_parent, parent_pos, 0).astype(np.int64)

    # réputation de l'auteur du parent : moyenne lissée sur le TRAIN
    author_code_u, _ = pd.factorize(union["author"], sort=False)
    gmean = float(np.nanmean(ups_u[:n_tr]))
    tr_ac = author_code_u[:n_tr]
    agg = pd.DataFrame({"a": tr_ac, "y": ups_u[:n_tr]}).groupby("a")["y"].agg(["sum", "count"])
    nca = int(author_code_u.max()) + 1
    s = np.nan_to_num(agg["sum"].reindex(np.arange(nca)).to_numpy())
    c = np.nan_to_num(agg["count"].reindex(np.arange(nca)).to_numpy())
    author_mean = (s + smoothing * gmean) / (c + smoothing)
    parent_author = np.where(has_parent, author_code_u[pp], -1)
    pa_mean = np.where(has_parent, author_mean[np.where(parent_author >= 0, parent_author, 0)], gmean)
    pa_count = np.where(has_parent, c[np.where(parent_author >= 0, parent_author, 0)], 0.0)

    out = pd.DataFrame({
        "id": union["id"].to_numpy(),
        "parent_author_mean": pa_mean.astype(np.float32),
        "parent_author_count_log": np.log1p(pa_count).astype(np.float32),
    })
    return out.iloc[:n_tr].reset_index(drop=True), out.iloc[n_tr:].reset_index(drop=True)
