"""Gradient boosting — cheval de bataille (CPU, LightGBM).

Entraîné avec un objectif adapté à la MAE (``mae`` L1, ``huber`` ou ``quantile`` alpha=0.5).
Évaluation de référence = **holdout temporel** (7 derniers jours) ; la soumission test est
produite en ré-entraînant sur tout le train avec le nombre d'arbres optimal trouvé.

Option ``log_target`` : entraîner sur un **log signé** ``sign(y)*log1p(|y|)`` puis la
transformation inverse — la médiane est préservée par la monotonie, et ``ups`` pouvant être
négatif (min observé -333, cf. docs/eda_findings.md), un ``log1p`` non signé produirait des NaN.
On compare empiriquement au MAE direct (résultat : MAE direct gagne largement ici, cf.
docs/guide.md — la cible se comporte mieux avec une perte L1 pure qu'après compression log).
"""
from __future__ import annotations

import numpy as np


def _signed_log1p(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.log1p(np.abs(y))


def _signed_expm1(t: np.ndarray) -> np.ndarray:
    return np.sign(t) * np.expm1(np.abs(t))

# Colonnes méta (jamais des features du modèle) : ids, codes de groupe, cible.
META_COLS = {"id", "created_utc", "link_id", "link_code", "author_code", "author_name", "ups"}
CATEGORICAL = ["hour", "dow"]


def feature_columns(df) -> list[str]:
    """Colonnes utilisées comme features (toutes sauf méta)."""
    return [c for c in df.columns if c not in META_COLS]


def _prepare_X(df, cols):
    """Cast bool->int8 pour LightGBM ; renvoie une copie légère des colonnes features."""
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype("int8")
    return X


def fit_predict(
    df_fit, df_val, df_test, cols, y_col: str, params: dict,
    early_stopping_rounds: int = 200, log_target: bool = False,
):
    """Entraîne sur df_fit, early-stopping sur df_val, prédit val et test.

    Retourne (val_pred, test_pred, model, best_iteration) — prédictions en espace ORIGINAL.
    """
    import lightgbm as lgb

    Xf, Xv = _prepare_X(df_fit, cols), _prepare_X(df_val, cols)
    yf = df_fit[y_col].to_numpy(dtype=float)
    yv = df_val[y_col].to_numpy(dtype=float)
    tf = _signed_log1p(yf) if log_target else yf
    tv = _signed_log1p(yv) if log_target else yv

    cat = [c for c in CATEGORICAL if c in cols]
    dtrain = lgb.Dataset(Xf, label=tf, categorical_feature=cat, free_raw_data=False)
    dvalid = lgb.Dataset(Xv, label=tv, reference=dtrain)
    model = lgb.train(
        params, dtrain,
        num_boost_round=params.get("num_iterations", 3000),
        valid_sets=[dvalid], valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                   lgb.log_evaluation(period=200)],
    )
    best = model.best_iteration or params.get("num_iterations", 3000)

    def _predict(X):
        p = model.predict(X, num_iteration=best)
        return _signed_expm1(p) if log_target else p

    val_pred = _predict(Xv)
    test_pred = _predict(_prepare_X(df_test, cols)) if df_test is not None else None
    return val_pred, test_pred, model, best


def train_full_predict(df_all, df_test, cols, y_col, params, num_rounds, log_target=False):
    """Ré-entraîne sur tout le train (nb d'arbres fixé) et prédit le test. Espace original."""
    import lightgbm as lgb

    X = _prepare_X(df_all, cols)
    y = df_all[y_col].to_numpy(dtype=float)
    t = _signed_log1p(y) if log_target else y
    cat = [c for c in CATEGORICAL if c in cols]
    dtrain = lgb.Dataset(X, label=t, categorical_feature=cat)
    p = dict(params); p.pop("num_iterations", None)
    model = lgb.train(p, dtrain, num_boost_round=int(num_rounds))
    pred = model.predict(_prepare_X(df_test, cols))
    return (_signed_expm1(pred) if log_target else pred), model


def lgb_params(cfg_gbm: dict, seed: int) -> dict:
    """Construit les hyperparamètres LightGBM depuis la config."""
    objective = cfg_gbm.get("objective", "mae")
    params = {
        "objective": objective,
        "metric": "mae",
        "learning_rate": cfg_gbm.get("learning_rate", 0.05),
        "num_leaves": cfg_gbm.get("num_leaves", 255),
        "num_iterations": cfg_gbm.get("n_estimators", 3000),
        "min_data_in_leaf": cfg_gbm.get("min_data_in_leaf", 200),
        "feature_fraction": cfg_gbm.get("feature_fraction", 0.8),
        "bagging_fraction": cfg_gbm.get("bagging_fraction", 0.8),
        "bagging_freq": 1,
        "seed": seed,
        "num_threads": cfg_gbm.get("num_threads", 0),
        "verbosity": -1,
    }
    if objective == "quantile":
        params["alpha"] = cfg_gbm.get("alpha", 0.5)
    if objective == "huber":
        params["alpha"] = cfg_gbm.get("huber_alpha", 1.0)
    return params
