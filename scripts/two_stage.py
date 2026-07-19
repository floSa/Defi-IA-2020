"""Modèle en deux étages, motivé par la forme de la cible (52% des `ups` valent exactement 1).

Idée : la MAE est minimisée par la médiane conditionnelle. Avec une masse de proba énorme sur
la valeur 1, un régresseur L1 unique passe son temps à arbitrer entre « c'est un 1 » et « c'est
une valeur de queue », et lisse les deux. On sépare explicitement les deux questions :

    Étage A : classifieur binaire  P(ups == 1 | x)
    Étage B : régresseur L1 entraîné UNIQUEMENT sur les lignes où ups != 1
    Prédiction : 1.0 si P >= seuil, sinon la sortie de l'étage B

Le seuil est choisi sur le holdout temporel (celui qui minimise la MAE), pas fixé a priori.
Référence à battre : la MAE du modèle à un étage, lue dans reports/gbm_*.json (jamais codée
en dur : une valeur figée devient fausse dès que le jeu de features change).
"""
import json
from pathlib import Path
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
auc = roc_auc_score((yv == 1).astype(int), p1)
print(f"[2stage] étage A : AUC={auc:.4f} "
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

# --- Combinaison : le seuil est un paramètre appris, il ne peut pas être réglé sur le jeu qui
# sert à le juger. On coupe le holdout en deux moitiés TEMPORELLES : on règle le seuil sur la
# première (tune), on rapporte la MAE sur la seconde (report), jamais vue par le réglage.
half = len(yv) // 2
tune, report = slice(None, half), slice(half, None)
GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.60, 0.70, 0.80, 0.90, 1.1]

results = {}
for t in GRID:
    results[round(t, 2)] = float(np.abs(np.where(p1[tune] >= t, 1.0, pb[tune]) - yv[tune]).mean())
best_t = min(results, key=results.get)

mae_report = float(np.abs(np.where(p1[report] >= best_t, 1.0, pb[report]) - yv[report]).mean())
mae_full = float(np.abs(np.where(p1 >= best_t, 1.0, pb) - yv).mean())

print("[2stage] MAE par seuil (moitié 'tune' du holdout) :", flush=True)
for t, m in sorted(results.items()):
    flag = "  <-- retenu" if t == best_t else ""
    label = {0.0: " (tout=1)", 1.1: " (jamais 1 => étage B seul)"}.get(t, "")
    print(f"    seuil {t:<5} MAE={m:.4f}{label}{flag}", flush=True)

print(f"\n[2stage] seuil retenu={best_t} (réglé sur la 1re moitié)", flush=True)
print(f"[2stage] MAE sur la 2e moitié, JAMAIS vue par le réglage : {mae_report:.4f}", flush=True)
# La référence à un étage se lit dans le dernier rapport GBM plutôt que codée en dur : une
# valeur figée devient silencieusement fausse dès que le jeu de features change, et affiche
# alors un gain gonflé (elle a annoncé -0.057 au lieu de -0.027 après le passage à l'hybride).
ref_path = Path("reports/gbm_clean_full.json")
ref = json.load(open(ref_path))["holdout"]["mae"] if ref_path.exists() else None
delta = f" | référence 1 étage = {ref:.4f} | delta={mae_full-ref:+.4f}" if ref else \
        " (référence 1 étage introuvable : lance d'abord train-gbm)"
print(f"[2stage] MAE sur le holdout entier (comparable au 1 étage) : {mae_full:.4f}{delta}",
      flush=True)
json.dump({"mae_by_threshold_tune_half": results, "best_threshold": best_t,
           "mae_report_half_unseen": mae_report, "mae_full_holdout": mae_full,
           "reference_single_stage": ref, "stage_a_auc": float(auc),
           # Consommés par two_stage_submit.py : le nombre d'arbres optimal dépend du jeu de
           # features et doit voyager avec lui, jamais être recopié en dur dans l'autre script.
           "stage_a_best_iter": int(clf.best_iteration),
           "stage_b_best_iter": int(reg.best_iteration),
           "n_fit_rows": int(len(fit_idx))},
          open("reports/two_stage.json", "w"), indent=2)
print(f"[2stage] terminé en {(time.time()-t0)/60:.1f} min -> reports/two_stage.json", flush=True)
