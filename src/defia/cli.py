"""Point d'entrée CLI du pipeline Défi IA 2020.

Chaque sous-commande correspond à une étape du plan (docs/plan.md §6) et est appelée par le
Makefile. La CLI charge la config YAML puis délègue aux modules du package. Les étapes de
modélisation lèvent ``NotImplementedError`` tant que le plan n'est pas validé — c'est
volontaire : le squelette est en place, l'implémentation suit la validation.

Usage :
    python -m defia.cli <étape> --config configs/default.yaml
"""
from __future__ import annotations

import click

from defia.config import load_config


@click.group()
def main() -> None:
    """Défi IA 2020 — prédiction des upvotes Reddit (MAE)."""


def _cfg(config: str):
    cfg = load_config(config)
    click.echo(f"[defia] config: {cfg.path}")
    return cfg


@main.command()
@click.option("--config", default="configs/default.yaml")
def data(config: str) -> None:
    """Raw CSV -> data/interim/{train,test}.parquet."""
    cfg = _cfg(config)
    from defia.data.load import build_train_test

    counts = build_train_test(cfg.resolve("raw_csv"), cfg.resolve("interim"))
    click.echo(f"[data] écrit train={counts['train']:,} test={counts['test']:,} "
               f"-> {cfg.resolve('interim')}")


@main.command()
@click.option("--config", default="configs/default.yaml")
def baseline(config: str) -> None:
    """Baselines MAE (médiane globale + médianes par groupe) en HOLDOUT TEMPOREL.

    Le split officiel étant temporel (train 1-24 mai, test 25-31 mai), on valide sur les 7
    derniers jours du train : c'est le juge de paix. On rapporte aussi le GroupKFold à titre de
    diagnostic (optimiste sur ce problème).
    """
    import json

    import numpy as np
    import pandas as pd

    from defia.data.load import load_split
    from defia.evaluation.cv import group_kfold_indices, temporal_holdout_indices
    from defia.evaluation.metrics import mae
    from defia.models.baseline import predict_group_median

    cfg = _cfg(config)
    interim = cfg.resolve("interim")
    target = cfg["data"]["target"]
    group = cfg["cv"]["group"]
    val_days = int(cfg["cv"].get("val_days", 7))

    click.echo("[baseline] chargement du train...")
    tr = load_split(interim, "train", columns=["id", target, group, "created_utc", "author"])
    tr["hour"] = (tr["created_utc"] // 3600) % 24
    y = tr[target].to_numpy(dtype=float)
    n = len(tr)
    click.echo(f"[baseline] {n:,} lignes ; médiane={np.median(y):g} moyenne={y.mean():.3f}")

    def eval_baselines(fit_df, val_df) -> dict[str, float]:
        yv = val_df[target].to_numpy(dtype=float)
        med = float(fit_df[target].median())
        preds = {
            "global_median": np.full(len(val_df), med),
            "median_by_hour": predict_group_median(fit_df, val_df, "hour", target),
            "median_by_author": predict_group_median(fit_df, val_df, "author", target),
            "const_1": np.ones(len(val_df)),
            "const_mean": np.full(len(val_df), fit_df[target].mean()),
        }
        return {k: mae(yv, p) for k, p in preds.items()}

    # --- Juge de paix : holdout temporel (7 derniers jours) ---
    fit_idx, val_idx = temporal_holdout_indices(tr["created_utc"].to_numpy(), val_days)
    temporal = eval_baselines(tr.iloc[fit_idx], tr.iloc[val_idx])
    click.echo(f"\n[baseline] MAE — HOLDOUT TEMPOREL ({val_days} derniers jours, "
               f"fit={len(fit_idx):,} / val={len(val_idx):,}) :")
    for name, val in sorted(temporal.items(), key=lambda kv: kv[1]):
        click.echo(f"   {name:20s} {val:.4f}")

    # --- Diagnostic : GroupKFold thread (optimiste) ---
    oof = {k: np.zeros(n) for k in ("global_median", "median_by_hour", "median_by_author")}
    for tri, vai in group_kfold_indices(tr[group].to_numpy(), 5, cfg.seed):
        f, v = tr.iloc[tri], tr.iloc[vai]
        oof["global_median"][vai] = float(f[target].median())
        oof["median_by_hour"][vai] = predict_group_median(f, v, "hour", target)
        oof["median_by_author"][vai] = predict_group_median(f, v, "author", target)
    groupkf = {k: mae(y, p) for k, p in oof.items()}
    click.echo("\n[baseline] MAE — GroupKFold thread (diagnostic, optimiste) :")
    for name, val in sorted(groupkf.items(), key=lambda kv: kv[1]):
        click.echo(f"   {name:20s} {val:.4f}")

    reports = cfg.resolve("reports"); reports.mkdir(parents=True, exist_ok=True)
    (reports / "baseline_mae.json").write_text(
        json.dumps({"temporal_holdout": temporal, "group_kfold": groupkf}, indent=2),
        encoding="utf-8")

    # Première soumission : médiane globale du train pour tout le test (format id,predicted)
    te = load_split(interim, "test", columns=["id"])
    best_const = float(np.median(y))
    sub = pd.DataFrame({"id": te["id"], "predicted": best_const})
    subs = cfg.resolve("submissions"); subs.mkdir(parents=True, exist_ok=True)
    out = subs / "submission_baseline_median.csv"
    sub.to_csv(out, index=False)
    click.echo(f"\n[baseline] submission : {out} ({len(sub):,} lignes, predicted={best_const:g})")


@main.command()
@click.option("--config", default="configs/default.yaml")
def features(config: str) -> None:
    """Construit les matrices de features (réseau + texte)."""
    _cfg(config)
    raise SystemExit("Étape 'features' à implémenter après validation du plan (docs/plan.md).")


@main.command("train-gbm")
@click.option("--config", default="configs/default.yaml")
def train_gbm(config: str) -> None:
    """Entraîne le GBM (objectif MAE) en GroupKFold thread."""
    _cfg(config)
    raise SystemExit("Étape 'train-gbm' à implémenter après validation du plan.")


@main.command()
@click.option("--config", default="configs/default.yaml")
def embeddings(config: str) -> None:
    """Extrait les embeddings de phrase (étape C, GPU)."""
    _cfg(config)
    raise SystemExit("Étape 'embeddings' à implémenter après validation du plan (GPU).")


@main.command("train-transformer")
@click.option("--config", default="configs/default.yaml")
def train_transformer(config: str) -> None:
    """Fine-tune l'encodeur body->ups (étape D, GPU)."""
    _cfg(config)
    raise SystemExit("Étape 'train-transformer' à implémenter après validation du plan (GPU).")


@main.command()
@click.option("--config", default="configs/default.yaml")
def blend(config: str) -> None:
    """Blending OOF des modèles -> prédiction finale."""
    _cfg(config)
    raise SystemExit("Étape 'blend' à implémenter après validation du plan.")


@main.command()
@click.option("--config", default="configs/default.yaml")
def submit(config: str) -> None:
    """Écrit submissions/submission.csv (id,predicted)."""
    _cfg(config)
    raise SystemExit("Étape 'submit' à implémenter après validation du plan.")


if __name__ == "__main__":
    main()
