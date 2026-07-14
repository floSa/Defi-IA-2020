"""Test CatBoost — catégorielles haute cardinalité natives (auteur, thread) + objectif MAE.

CatBoost gère nativement author_code / link_code (encodage CTR ordonné, sans fuite) là où
LightGBM doit les ignorer. Mesure la MAE sur le holdout temporel vs le champion GBM (8.2667).
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from defia.evaluation.cv import temporal_holdout_indices
from defia.evaluation.metrics import mae

P = "data/processed"
SAMPLE = 1_200_000  # sous-échantillonne le fit pour itérer vite (comparable aux runs GBM --sample)
tr = pd.read_parquet(f"{P}/train_features.parquet")
for name in ["author_enc", "context", "author_dyn"]:  # graph abandonné (n'apporte rien)
    try:
        tr = tr.merge(pd.read_parquet(f"{P}/train_{name}.parquet"), on="id", how="left")
    except FileNotFoundError:
        pass
print("features chargées:", tr.shape)

CAT = [c for c in ["author_code", "hour", "dow"] if c in tr.columns]  # link_code retiré (OOM)
DROP = {"id", "created_utc", "link_id", "author_name", "ups"}
feat = [c for c in tr.columns if c not in DROP]
for c in CAT:
    tr[c] = tr[c].astype("int64").astype("str")  # CatBoost veut des str/int catégoriels
# bool -> int
for c in feat:
    if tr[c].dtype == bool:
        tr[c] = tr[c].astype("int8")

fit_idx, val_idx = temporal_holdout_indices(tr["created_utc"].to_numpy(), 7)
rng = np.random.default_rng(42)
if len(fit_idx) > SAMPLE:
    fit_idx = np.sort(rng.choice(fit_idx, size=SAMPLE, replace=False))
Xf, yf = tr.iloc[fit_idx][feat], tr.iloc[fit_idx]["ups"].to_numpy(float)
Xv, yv = tr.iloc[val_idx][feat], tr.iloc[val_idx]["ups"].to_numpy(float)
cat_idx = [feat.index(c) for c in CAT]
print("cat_features:", CAT, "| fit:", len(fit_idx))

t0 = time.time()
model = CatBoostRegressor(
    loss_function="MAE", iterations=800, learning_rate=0.1, depth=8,
    task_type="CPU", thread_count=-1, random_seed=42, verbose=200,
    l2_leaf_reg=3.0,
)
model.fit(Pool(Xf, yf, cat_features=cat_idx), eval_set=Pool(Xv, yv, cat_features=cat_idx),
          use_best_model=True, early_stopping_rounds=150)
pred = model.predict(Xv)
print(f"\n[catboost] MAE holdout = {mae(yv, pred):.4f}  (best_iter={model.best_iteration_}, "
      f"{time.time()-t0:.0f}s)")
imp = pd.Series(model.feature_importances_, index=feat).sort_values(ascending=False)
print("[catboost] top features:\n", imp.head(15).to_string())
