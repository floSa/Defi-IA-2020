"""Soumission du modèle deux étages (cf. scripts/two_stage.py pour la validation du principe).

Produit trois artefacts, alignés sur les conventions du projet pour que `defia blend` les voie :
    submissions/submission_two_stage.csv
    data/processed/oof/oof_two_stage.parquet   (prédictions sur le holdout temporel)
    data/processed/oof/test_two_stage.parquet  (prédictions test)

Le nombre d'arbres de chaque étage et le seuil de combinaison sont ceux trouvés lors de la
validation ; le ré-entraînement final se fait sur tout le train (le cap mémoire adaptatif de
cli.py laisse passer les 3,2M lignes sur cette machine).
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
oof_dir = processed / "oof"; oof_dir.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.30          # optimum intérieur, courbe plate 0.25-0.45
PARAMS = {"learning_rate": 0.05, "num_leaves": 255, "verbose": -1, "num_threads": 10}
t0 = time.time()


def load(split: str) -> pd.DataFrame:
    df = pd.read_parquet(processed / f"{split}_features.parquet")
    for name in ["author_enc", "emb", "context", "author_dyn", "parentenc", "interactions"]:
        p = processed / f"{split}_{name}.parquet"
        if p.exists():
            df = df.merge(pd.read_parquet(p), on="id", how="left")
    return df


tr, te = load("train"), load("test")
cols = feature_columns(tr)
cols = [c for c in cols if c in te.columns]
print(f"[2stage-sub] {len(cols)} features | train={len(tr):,} test={len(te):,} "
      f"({time.time()-t0:.0f}s)", flush=True)

fit_idx, val_idx = temporal_holdout_indices(tr["created_utc"].to_numpy(),
                                            int(cfg["cv"].get("val_days", 7)))
val_ids = tr["id"].to_numpy()[val_idx]
X = tr[cols].astype(np.float32)
y = tr["ups"].to_numpy(dtype=float)
Xte = te[cols].astype(np.float32)
te_ids = te["id"].to_numpy()
del tr, te


def two_stage(Xf, yf, Xp, n_a: int, n_b: int) -> np.ndarray:
    """Entraîne les deux étages sur (Xf, yf) et prédit sur Xp."""
    clf = lgb.train({**PARAMS, "objective": "binary"},
                    lgb.Dataset(Xf, label=(yf == 1).astype(int)), num_boost_round=n_a)
    mask = yf != 1
    reg = lgb.train({**PARAMS, "objective": "mae"},
                    lgb.Dataset(Xf[mask], label=yf[mask]), num_boost_round=n_b)
    return np.where(clf.predict(Xp) >= THRESHOLD, 1.0, reg.predict(Xp))


# --- 1. OOF sur le holdout (entraîné sans le holdout, comme les autres modèles du blend) ---
# Le nombre d'arbres optimal dépend du jeu de features : il se lit dans le rapport produit par
# two_stage.py, jamais recopié en dur. Les valeurs figées 542/1050 dataient des embeddings de la
# première session et faisaient tourner l'étage B avec 70% d'arbres en trop sur la base propre.
ts = json.load(open("reports/two_stage.json"))
N_A, N_B = ts["stage_a_best_iter"], ts["stage_b_best_iter"]
print(f"[2stage-sub] arbres lus dans reports/two_stage.json : etage A {N_A}, etage B {N_B}",
      flush=True)
print("[2stage-sub] OOF holdout...", flush=True)
oof_pred = two_stage(X.iloc[fit_idx], y[fit_idx], X.iloc[val_idx], N_A, N_B)
oof_mae = float(np.abs(oof_pred - y[val_idx]).mean())
print(f"[2stage-sub] MAE holdout={oof_mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame({"id": val_ids, "pred": oof_pred}).to_parquet(
    oof_dir / "oof_two_stage.parquet", index=False)

# --- 2. Ré-entraînement sur tout le train -> prédictions test ---
# Les étages ont convergé sur la portion « fit » ; on met à l'échelle du train complet pour
# garder une capacité comparable (pas d'early stopping possible sans jeu de validation).
scale = len(X) / len(fit_idx)
n_a, n_b = int(N_A * scale), int(N_B * scale)
print(f"[2stage-sub] ré-entraînement full train ({len(X):,} lignes, "
      f"{n_a} + {n_b} arbres)...", flush=True)
test_pred = np.clip(two_stage(X, y, Xte, n_a, n_b), -50, None)

subs = cfg.resolve("submissions"); subs.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"id": te_ids, "predicted": test_pred}).to_csv(
    subs / "submission_two_stage.csv", index=False)
pd.DataFrame({"id": te_ids, "pred": test_pred}).to_parquet(
    oof_dir / "test_two_stage.parquet", index=False)
json.dump({"mae_holdout": oof_mae, "threshold": THRESHOLD,
           "n_trees_stage_a": n_a, "n_trees_stage_b": n_b, "n_features": len(cols)},
          open("reports/gbm_two_stage.json", "w"), indent=2)
print(f"[2stage-sub] écrit submission_two_stage.csv + OOF ({(time.time()-t0)/60:.1f} min)",
      flush=True)
