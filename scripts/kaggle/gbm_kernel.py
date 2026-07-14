"""Kernel Kaggle (CPU, ~30 Go RAM) — LightGBM sur TOUTES les features, sans contrainte mémoire.

Contourne le mur des 7,4 Go du laptop : ici on charge toutes les matrices de features déjà
calculées (attachées comme dataset Kaggle `defia-features`), on les fusionne, et on entraîne le
LightGBM complet (96 features, tout le train, sans sous-échantillonnage) avec l'objectif MAE.

Sorties (/kaggle/working) : oof_gbm_kaggle.parquet, test_gbm_kaggle.parquet, submission_gbm_kaggle.csv,
metrics.json (MAE holdout temporel).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb

def _find_input():
    print("[kernel] /kaggle/input :", glob.glob("/kaggle/input/*"))
    hits = glob.glob("/kaggle/input/**/train_features.parquet", recursive=True)
    if not hits:
        raise FileNotFoundError("train_features.parquet introuvable (dataset non attaché ?)")
    return os.path.dirname(hits[0])


IN = None  # résolu dans main() via _find_input()
OUT = "/kaggle/working"
VAL_DAYS = 7
META = {"id", "created_utc", "link_id", "link_code", "author_code", "author_name", "ups"}
BLOCKS = ["author_enc", "tfidf", "context", "author_dyn", "parentenc", "interactions"]


def load(split):
    global IN
    if IN is None:
        IN = _find_input()
        print("[kernel] input_dir =", IN)
    df = pd.read_parquet(f"{IN}/{split}_features.parquet")
    for b in BLOCKS:
        p = f"{IN}/{split}_{b}.parquet"
        if os.path.exists(p):
            df = df.merge(pd.read_parquet(p), on="id", how="left")
    return df


def main():
    tr, te = load("train"), load("test")
    cols = [c for c in tr.columns if c not in META]
    for c in cols:
        if tr[c].dtype == bool:
            tr[c] = tr[c].astype("int8"); te[c] = te[c].astype("int8")
    print(f"[kernel] {len(cols)} features, train={len(tr):,}, test={len(te):,}")

    cutoff = int(tr["created_utc"].max()) - VAL_DAYS * 86_400
    fit = tr[tr.created_utc <= cutoff]; val = tr[tr.created_utc > cutoff]
    cat = [c for c in ["hour", "dow"] if c in cols]
    params = dict(objective="mae", metric="mae", learning_rate=0.05, num_leaves=255,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, seed=42, num_threads=os.cpu_count(), verbosity=-1)

    dtr = lgb.Dataset(fit[cols], fit["ups"], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(val[cols], val["ups"], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(200)])
    best = model.best_iteration
    val_pred = model.predict(val[cols], num_iteration=best)
    mae = float(np.mean(np.abs(val["ups"].to_numpy() - val_pred)))
    print(f"[kernel] HOLDOUT MAE={mae:.4f} best_iter={best}")

    pd.DataFrame({"id": val["id"].to_numpy(), "pred": val_pred}).to_parquet(f"{OUT}/oof_gbm_kaggle.parquet", index=False)

    # ré-entraînement sur tout le train (30 Go -> aucun souci mémoire)
    dall = lgb.Dataset(tr[cols], tr["ups"], categorical_feature=cat)
    full = lgb.train(params, dall, num_boost_round=int(best))
    test_pred = np.clip(full.predict(te[cols]), -50, None)
    pd.DataFrame({"id": te["id"].to_numpy(), "pred": test_pred}).to_parquet(f"{OUT}/test_gbm_kaggle.parquet", index=False)
    pd.DataFrame({"id": te["id"], "predicted": test_pred}).to_csv(f"{OUT}/submission_gbm_kaggle.csv", index=False)
    json.dump({"holdout_mae": mae, "n_features": len(cols), "best_iter": int(best)},
              open(f"{OUT}/metrics.json", "w"), indent=2)
    print("[kernel] terminé.")


if __name__ == "__main__":
    main()
