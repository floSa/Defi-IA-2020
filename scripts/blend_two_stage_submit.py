"""Soumission du meilleur modèle : blend + règle deux étages par-dessus (MAE holdout 7.9276).

Réutilise les prédictions test du blend déjà écrites par `defia blend` et leur applique la
règle « si P(ups==1) >= seuil, réponds 1 », le classifieur étant ré-entraîné sur tout le train.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "src")
from defia.config import load_config
from defia.models.gbm import feature_columns

cfg = load_config("configs/default.yaml")
processed = cfg.resolve("processed")
subs = cfg.resolve("submissions")
THRESHOLD = json.load(open("reports/blend_two_stage.json"))["best_threshold"]
t0 = time.time()


def load_features(split: str) -> pd.DataFrame:
    df = pd.read_parquet(processed / f"{split}_features.parquet")
    for name in ["author_enc", "emb", "context", "author_dyn", "parentenc", "interactions"]:
        p = processed / f"{split}_{name}.parquet"
        if p.exists():
            df = df.merge(pd.read_parquet(p), on="id", how="left")
    return df


tr, te = load_features("train"), load_features("test")
cols = [c for c in feature_columns(tr) if c in te.columns]
y = tr["ups"].to_numpy(dtype=float)
X = tr[cols].astype(np.float32)
Xte, te_ids = te[cols].astype(np.float32), te["id"].to_numpy()
del tr, te
print(f"[sub] {len(cols)} features, seuil={THRESHOLD} ({time.time()-t0:.0f}s)", flush=True)

# 542 arbres avaient suffi sur 71% des lignes ; mise à l'échelle pour le train complet.
n_trees = int(542 * (len(X) / (len(X) * 0.71)))
print(f"[sub] classifieur ups==1 sur tout le train ({len(X):,} lignes, {n_trees} arbres)...",
      flush=True)
clf = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 255,
                 "verbose": -1, "num_threads": 10},
                lgb.Dataset(X, label=(y == 1).astype(int)), num_boost_round=n_trees)
p1 = clf.predict(Xte)

blend = pd.read_csv(subs / "submission_final_clean.csv")
assert (blend["id"].to_numpy() == te_ids).all(), "ids désalignés entre blend et features test"
final = np.where(p1 >= THRESHOLD, 1.0, blend["predicted"].to_numpy())
out = subs / "submission_final_clean2s.csv"
pd.DataFrame({"id": te_ids, "predicted": final}).to_csv(out, index=False)
print(f"[sub] part des prédictions forcées à 1 : {(p1 >= THRESHOLD).mean():.1%}", flush=True)
print(f"[sub] écrit {out} ({(time.time()-t0)/60:.1f} min)", flush=True)
