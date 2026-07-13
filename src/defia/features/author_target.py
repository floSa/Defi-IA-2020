"""Target encoding temporellement propre pour l'auteur (utilise ``ups`` — attention à la fuite).

Contrairement à ``structural.py`` (transductif, sans label), ce module utilise la cible et doit
donc respecter scrupuleusement l'ordre temporel :

  * TRAIN : moyenne lissée (bayésienne) des ``ups`` des commentaires **strictement antérieurs**
    du même auteur (expanding, exclut la ligne courante elle-même).
  * TEST : moyenne lissée calculée sur la **totalité** de l'historique auteur observé en train —
    légitime car le split est temporel sans recouvrement (tout le train précède tout le test,
    cf. docs/eda_findings.md).

Le lissage bayésien (paramètre ``smoothing`` = nb de pseudo-observations à la moyenne globale)
évite la dégradation observée avec un encodage naïf en Milestone A (médiane/auteur : MAE 13.25,
pire que la baseline 11.91 — cf. docs/guide.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_author_history_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = "ups",
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attend train_df/test_df avec colonnes id, author, created_utc (+ ``target`` pour train).

    Retourne (train_feat, test_feat), chacun avec : id, author_hist_mean, author_hist_count_log.
    """
    global_mean = float(train_df[target].mean())

    # Codes auteur cohérents entre train et test (mêmes entiers pour le même auteur).
    all_authors = pd.concat([train_df["author"], test_df["author"]], ignore_index=True)
    codes, _ = pd.factorize(all_authors, sort=False)
    n_tr = len(train_df)
    tr_code, te_code = codes[:n_tr], codes[n_tr:]

    # --- TRAIN : expanding leave-one-out, strictement antérieur (tri par auteur puis temps) ---
    order = np.lexsort((train_df["created_utc"].to_numpy(), tr_code))
    y_sorted = train_df[target].to_numpy(dtype=float)[order]
    g_sorted = tr_code[order]
    df_sorted = pd.DataFrame({"g": g_sorted, "y": y_sorted})
    cum_sum = df_sorted.groupby("g")["y"].cumsum().to_numpy()
    cum_cnt = (df_sorted.groupby("g").cumcount() + 1).to_numpy()
    prior_sum = cum_sum - y_sorted   # exclut la ligne courante
    prior_cnt = cum_cnt - 1
    smoothed_sorted = (prior_sum + smoothing * global_mean) / (prior_cnt + smoothing)

    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))
    train_feat = pd.DataFrame({
        "id": train_df["id"].to_numpy(),
        "author_hist_mean": smoothed_sorted[inv].astype(np.float32),
        "author_hist_count_log": np.log1p(prior_cnt[inv]).astype(np.float32),
    })

    # --- TEST : agrégat complet du train par auteur (tout le train précède tout le test) ---
    agg = (pd.DataFrame({"g": tr_code, "y": train_df[target].to_numpy(dtype=float)})
           .groupby("g")["y"].agg(["sum", "count"]))
    n_codes = int(codes.max()) + 1 if len(codes) else 0
    sum_map = agg["sum"].reindex(np.arange(n_codes)).to_numpy()
    cnt_map = agg["count"].reindex(np.arange(n_codes)).to_numpy()
    te_sum = np.nan_to_num(sum_map[te_code], nan=0.0)
    te_cnt = np.nan_to_num(cnt_map[te_code], nan=0.0)
    te_mean = (te_sum + smoothing * global_mean) / (te_cnt + smoothing)

    test_feat = pd.DataFrame({
        "id": test_df["id"].to_numpy(),
        "author_hist_mean": te_mean.astype(np.float32),
        "author_hist_count_log": np.log1p(te_cnt).astype(np.float32),
    })
    return train_feat, test_feat
