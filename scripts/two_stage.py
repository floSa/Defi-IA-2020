"""Modèle en deux étages, motivé par la forme de la cible (52% des `ups` valent exactement 1).

Idée : la MAE est minimisée par la médiane conditionnelle. Avec une masse de proba énorme sur
la valeur 1, un régresseur L1 unique passe son temps à arbitrer entre « c'est un 1 » et « c'est
une valeur de queue », et lisse les deux. On sépare explicitement les deux questions :

    Étage A : classifieur binaire  P(ups == 1 | x)
    Étage B : régresseur L1 entraîné UNIQUEMENT sur les lignes où ups != 1
    Prédiction : 1.0 si P >= seuil, sinon la sortie de l'étage B

Le seuil est choisi sur le holdout temporel (celui qui minimise la MAE), pas fixé a priori.
Référence à battre : MAE 7.9968 (GBM 128 features, un seul étage).
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "src")
from defia.config import load_config
from defia.evaluation.cv import temporal_holdout_indices
from defia.models.gbm import feature_columns

cfg = load_config("configs/default.yaml")
processed = cfg.resolve("processed")
t0 = time.time()


def load_all() -> pd.DataFrame:
    """Reconstitue le même jeu de features que `defia train-gbm` (train uniquement)."""
    tr = pd.read_parquet(processed / "train_features.parquet")
    for name in ["author_enc", "emb", "context", "author_dyn", "parentenc", "interactions"]:
        p = processed / f"train_{name}.parquet"
        if p.exists():
            tr = tr.merge(pd.read_parquet(p), on="id", how="left")
    return tr


tr = load_all()
cols = feature_columns(tr)
print(f"[2stage] {len(cols)} features, {len(tr):,} lignes ({time.time()-t0:.0f}s)", flush=True)

fit_idx, val_idx = temporal_holdout_indices(tr["created_utc"].to_numpy(),
                                            int(cfg["cv"].get("val_days", 7)))
X = tr[cols].astype(np.float32)
y = tr["ups"].to_numpy(dtype=float)
del tr

Xf, Xv = X.iloc[fit_idx], X.iloc[val_idx]
yf, yv = y[fit_idx], y[val_idx]
print(f"[2stage] fit={len(Xf):,} val={len(Xv):,} | part de ups==1 : "
      f"{(yf == 1).mean():.1%} (fit) {(yv == 1).mean():.1%} (val)", flush=True)

# --- Étage A : P(ups == 1) ---
print("[2stage] étage A (classifieur ups==1)...", flush=True)
clf = lgb.train(
    {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 255,
     "verbose": -1, "num_threads": 10},
    lgb.Dataset(Xf, label=(yf == 1).astype(int)),
    num_boost_round=1500,
    valid_sets=[lgb.Dataset(Xv, label=(yv == 1).astype(int))],
    callbacks=[lgb.early_stopping(100, verbose=False)],
)
p1 = clf.predict(Xv, num_iteration=clf.best_iteration)
from sklearn.metrics import roc_auc_score
print(f"[2stage] étage A : AUC={roc_auc_score((yv == 1).astype(int), p1):.4f} "
      f"best_iter={clf.best_iteration} ({time.time()-t0:.0f}s)", flush=True)

# --- Étage B : régression L1 sur les non-1 seulement ---
print("[2stage] étage B (régression L1 sur ups != 1)...", flush=True)
mask_f, mask_v = yf != 1, yv != 1
reg = lgb.train(
    {"objective": "mae", "metric": "l1", "learning_rate": 0.05, "num_leaves": 255,
     "verbose": -1, "num_threads": 10},
    lgb.Dataset(Xf[mask_f], label=yf[mask_f]),
    num_boost_round=3000,
    valid_sets=[lgb.Dataset(Xv[mask_v], label=yv[mask_v])],
    callbacks=[lgb.early_stopping(200, verbose=False)],
)
pb = reg.predict(Xv, num_iteration=reg.best_iteration)
print(f"[2stage] étage B : best_iter={reg.best_iteration} ({time.time()-t0:.0f}s)", flush=True)

# --- Combinaison : seuil choisi sur le holdout ---
results = {}
for t in [0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.1]:
    pred = np.where(p1 >= t, 1.0, pb)
    results[round(t, 2)] = float(np.abs(pred - yv).mean())
best_t = min(results, key=results.get)
print("[2stage] MAE par seuil :", flush=True)
for t, m in sorted(results.items()):
    flag = "  <-- meilleur" if t == best_t else ""
    label = {0.0: " (tout=1)", 1.1: " (jamais 1 => étage B seul)"}.get(t, "")
    print(f"    seuil {t:<5} MAE={m:.4f}{label}{flag}", flush=True)

print(f"\n[2stage] MEILLEUR : seuil={best_t} MAE={results[best_t]:.4f} "
      f"| référence 1 étage = 7.9968 | delta={results[best_t]-7.9968:+.4f}", flush=True)
json.dump({"mae_by_threshold": results, "best_threshold": best_t,
           "best_mae": results[best_t], "reference_single_stage": 7.9968},
          open("reports/two_stage.json", "w"), indent=2)
print(f"[2stage] terminé en {(time.time()-t0)/60:.1f} min -> reports/two_stage.json", flush=True)
