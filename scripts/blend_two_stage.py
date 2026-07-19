"""Deux étages appliqué AU BLEND, et non à un seul modèle.

État actuel : le blend mélange deux régresseurs (7.9341). Le modèle deux étages, lui, applique
sa règle « si P(ups==1) est élevé, réponds 1 » à ses propres prédictions uniquement.

Idée testée ici : garder le meilleur régresseur possible (le blend) et lui appliquer la règle
par-dessus. Autrement dit, décider « c'est un 1 » séparément, et ne laisser le blend s'exprimer
que sur les commentaires dont on pense qu'ils sortent du lot.

    prédiction = 1.0 si P(ups==1) >= seuil, sinon prédiction du blend

Le seuil est réglé sur la première moitié temporelle du holdout et jugé sur la seconde, jamais
vue par le réglage (même protocole que scripts/two_stage.py).
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "src")
from defia.config import load_config
from defia.data.load import load_split
from defia.evaluation.cv import temporal_holdout_indices
from defia.models.gbm import feature_columns

cfg = load_config("configs/default.yaml")
processed = cfg.resolve("processed")
oof_dir = processed / "oof"
t0 = time.time()

# --- Vérité holdout + prédictions du blend, alignées sur les mêmes ids ---
tr_meta = load_split(cfg.resolve("interim"), "train", ["id", "ups", "created_utc"])
_, val_idx = temporal_holdout_indices(tr_meta["created_utc"].to_numpy(),
                                      int(cfg["cv"].get("val_days", 7)))
truth = tr_meta.iloc[val_idx][["id", "ups"]].reset_index(drop=True)

weights = json.load(open("reports/blend_final_hybrid.json"))["weights"]
weights = {k: w for k, w in weights.items() if w > 1e-6}
print(f"[blend2s] blend = {', '.join(f'{k} {w:.3f}' for k, w in weights.items())}", flush=True)

blend_pred = np.zeros(len(truth))
for name, w in weights.items():
    o = pd.read_parquet(oof_dir / f"oof_{name}.parquet")
    merged = truth[["id"]].merge(o, on="id", how="left")
    assert merged["pred"].notna().all(), f"OOF incomplet pour {name}"
    blend_pred += w * merged["pred"].to_numpy()
y = truth["ups"].to_numpy(dtype=float)
print(f"[blend2s] MAE du blend seul : {np.abs(blend_pred - y).mean():.4f} "
      f"({time.time()-t0:.0f}s)", flush=True)

# --- Étage A : le classifieur P(ups == 1), réentraîné hors holdout ---
def load_features(split: str) -> pd.DataFrame:
    df = pd.read_parquet(processed / f"{split}_features.parquet")
    for name in ["author_enc", "emb", "context", "author_dyn", "parentenc", "interactions"]:
        p = processed / f"{split}_{name}.parquet"
        if p.exists():
            df = df.merge(pd.read_parquet(p), on="id", how="left")
    return df


tr = load_features("train")
cols = feature_columns(tr)
fit_idx, val_idx2 = temporal_holdout_indices(tr["created_utc"].to_numpy(),
                                             int(cfg["cv"].get("val_days", 7)))
assert (tr["id"].to_numpy()[val_idx2] == truth["id"].to_numpy()).all(), "désalignement holdout"
X = tr[cols].astype(np.float32)
yf = tr["ups"].to_numpy(dtype=float)[fit_idx]
del tr

print("[blend2s] entraînement de l'étage A (classifieur ups==1)...", flush=True)
clf = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 255,
                 "verbose": -1, "num_threads": 10},
                lgb.Dataset(X.iloc[fit_idx], label=(yf == 1).astype(int)),
                num_boost_round=542)
p1 = clf.predict(X.iloc[val_idx2])
print(f"[blend2s] étage A prêt ({time.time()-t0:.0f}s)", flush=True)

# --- Balayage du seuil : réglage sur la 1re moitié, jugement sur la 2e ---
half = len(y) // 2
GRID = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 1.1]
tune = {round(t, 2): float(np.abs(np.where(p1[:half] >= t, 1.0, blend_pred[:half])
                                  - y[:half]).mean()) for t in GRID}
best_t = min(tune, key=tune.get)
mae_unseen = float(np.abs(np.where(p1[half:] >= best_t, 1.0, blend_pred[half:]) - y[half:]).mean())
mae_full = float(np.abs(np.where(p1 >= best_t, 1.0, blend_pred) - y).mean())
blend_unseen = float(np.abs(blend_pred[half:] - y[half:]).mean())
blend_full = float(np.abs(blend_pred - y).mean())

print("[blend2s] MAE par seuil (moitié de réglage) :", flush=True)
for t, m in sorted(tune.items()):
    label = {0.0: " (tout=1)", 1.1: " (blend seul)"}.get(t, "")
    print(f"    seuil {t:<5} MAE={m:.4f}{label}{'  <-- retenu' if t == best_t else ''}", flush=True)

print(f"\n[blend2s] seuil retenu = {best_t}", flush=True)
print(f"[blend2s] 2e moitié (jamais vue) : blend seul {blend_unseen:.4f} -> "
      f"blend + 2 étages {mae_unseen:.4f}  ({mae_unseen-blend_unseen:+.4f})", flush=True)
print(f"[blend2s] holdout entier         : blend seul {blend_full:.4f} -> "
      f"blend + 2 étages {mae_full:.4f}  ({mae_full-blend_full:+.4f})", flush=True)
json.dump({"mae_by_threshold_tune": tune, "best_threshold": best_t,
           "blend_alone_full": blend_full, "blend_two_stage_full": mae_full,
           "blend_alone_unseen_half": blend_unseen, "blend_two_stage_unseen_half": mae_unseen},
          open("reports/blend_two_stage.json", "w"), indent=2)
print(f"[blend2s] terminé en {(time.time()-t0)/60:.1f} min", flush=True)
