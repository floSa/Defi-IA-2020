"""Chargement du CSV brut et conversion train/test en parquet.

Points d'attention (issus de l'EDA, cf. docs/eda_findings.md) :
  * ``ups == NaN``  ->  ligne de test ; sinon train.
  * Le ``body`` contient du markdown brut et des entités HTML échappées (``&lt; &gt; &amp;``).
    On conserve le texte brut (le markdown est un signal) et on dérive une version nettoyée.
  * dtypes explicites pour tenir en RAM (4,2 M lignes) : ids en category, created_utc en int32.

TODO (post-validation du plan) : implémenter ``load_raw`` et ``build_train_test``.
"""
from __future__ import annotations

from pathlib import Path

# import pandas as pd  # activé à l'implémentation


def load_raw(csv_path: str | Path):
    """Charge ``comments_students.csv`` avec dtypes optimisés. Retourne un DataFrame."""
    raise NotImplementedError("À implémenter après validation du plan (docs/plan.md).")


def build_train_test(csv_path: str | Path, out_dir: str | Path):
    """Sépare train (ups renseigné) / test (ups NaN) et persiste en parquet."""
    raise NotImplementedError("À implémenter après validation du plan (docs/plan.md).")
