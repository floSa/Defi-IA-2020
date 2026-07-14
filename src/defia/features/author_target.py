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


def build_author_dynamics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = "ups",
    smoothing: float = 20.0,
    viral_thr: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Réputation d'auteur enrichie, temporellement propre (expanding sur le passé strict).

    Au-delà de la moyenne : écart-type, max, fraction virale (ups>seuil) et downvotée (ups<=0) de
    l'historique de l'auteur. Pour le train, statistiques sur les commentaires STRICTEMENT
    antérieurs du même auteur ; pour le test, sur tout l'historique train de l'auteur (split
    temporel). Retourne (train_feat, test_feat) avec id + colonnes author_dyn_*.
    """
    gmean = float(train_df[target].mean())
    all_a = pd.concat([train_df["author"], test_df["author"]], ignore_index=True)
    codes, _ = pd.factorize(all_a, sort=False)
    n_tr = len(train_df)
    tr_code, te_code = codes[:n_tr], codes[n_tr:]

    order = np.lexsort((train_df["created_utc"].to_numpy(), tr_code))
    y = train_df[target].to_numpy(dtype=float)[order]
    g = tr_code[order]
    viral = (y > viral_thr).astype(float)
    down = (y <= 0).astype(float)

    df = pd.DataFrame({"g": g, "y": y, "y2": y * y, "v": viral, "d": down})
    grp = df.groupby("g")
    csum = grp["y"].cumsum().to_numpy() - y
    csum2 = grp["y2"].cumsum().to_numpy() - y * y
    ccnt = grp.cumcount().to_numpy().astype(float)  # nb strictement antérieurs
    cmax = grp["y"].cummax().to_numpy()
    cvir = grp["v"].cumsum().to_numpy() - viral
    cdwn = grp["d"].cumsum().to_numpy() - down

    denom = np.maximum(ccnt, 1)
    mean = (csum + smoothing * gmean) / (ccnt + smoothing)
    var = np.maximum(csum2 / denom - (csum / denom) ** 2, 0)
    inv = np.empty(len(order), dtype=np.int64); inv[order] = np.arange(len(order))

    def _mk(ids, mean_, std_, max_, cnt_, vir_, dwn_):
        return pd.DataFrame({
            "id": ids,
            "author_dyn_mean": mean_.astype(np.float32),
            "author_dyn_std": std_.astype(np.float32),
            "author_dyn_max": max_.astype(np.float32),
            "author_dyn_count_log": np.log1p(cnt_).astype(np.float32),
            "author_dyn_viral_frac": vir_.astype(np.float32),
            "author_dyn_down_frac": dwn_.astype(np.float32),
        })

    # max strictement antérieur : cummax décalé d'un cran dans le groupe (exclut la ligne courante)
    shifted = pd.Series(cmax).groupby(g).shift(1).to_numpy()
    prior_max = np.where(np.isnan(shifted), gmean, shifted)
    train_feat = _mk(
        train_df["id"].to_numpy(),
        mean[inv], np.sqrt(var)[inv], prior_max[inv],
        ccnt[inv], (cvir / (ccnt + smoothing))[inv], (cdwn / (ccnt + smoothing))[inv])

    # TEST : agrégat complet du train par auteur
    yt = train_df[target].to_numpy(dtype=float)
    agg = pd.DataFrame({"g": tr_code, "y": yt, "v": (yt > viral_thr).astype(float),
                        "d": (yt <= 0).astype(float)}).groupby("g").agg(
        s=("y", "sum"), s2=("y", lambda x: float((x * x).sum())), c=("y", "count"),
        mx=("y", "max"), vv=("v", "sum"), dd=("d", "sum"))
    nc = int(codes.max()) + 1 if len(codes) else 0
    idx = np.arange(nc)
    s = np.nan_to_num(agg["s"].reindex(idx).to_numpy())
    s2 = np.nan_to_num(agg["s2"].reindex(idx).to_numpy())
    c = np.nan_to_num(agg["c"].reindex(idx).to_numpy())
    mx = agg["mx"].reindex(idx).to_numpy()
    vv = np.nan_to_num(agg["vv"].reindex(idx).to_numpy())
    dd = np.nan_to_num(agg["dd"].reindex(idx).to_numpy())
    te_c = c[te_code]; te_s = s[te_code]
    te_mean = (te_s + smoothing * gmean) / (te_c + smoothing)
    te_var = np.maximum(np.where(te_c > 0, s2[te_code] / np.maximum(te_c, 1) - (te_s / np.maximum(te_c, 1)) ** 2, 0), 0)
    te_max = np.where(np.isnan(mx[te_code]), gmean, mx[te_code])
    test_feat = _mk(test_df["id"].to_numpy(), te_mean, np.sqrt(te_var), te_max,
                    te_c, vv[te_code] / (te_c + smoothing), dd[te_code] / (te_c + smoothing))
    return train_feat, test_feat


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
