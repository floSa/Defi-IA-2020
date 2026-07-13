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

    build_train_test(cfg.resolve("raw_csv"), cfg.resolve("interim"))


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
