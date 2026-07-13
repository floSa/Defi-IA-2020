"""Chargement et accès à la configuration YAML du pipeline.

La config est un simple dictionnaire chargé depuis ``configs/*.yaml`` ; on l'expose via une
petite dataclass pour l'auto-complétion et la validation des chemins. Toutes les étapes du
pipeline reçoivent cet objet, ce qui rend le code agnostique à l'environnement d'exécution
(local CPU, desktop GPU, Kaggle, Colab) : seules les valeurs YAML changent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Racine du repo : deux niveaux au-dessus de ce fichier (src/defia/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Vue typée d'un fichier de configuration YAML."""

    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # --- accès pratique aux sections courantes ---
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    def resolve(self, path_key: str) -> Path:
        """Résout un chemin de la section ``paths`` relativement à la racine du repo."""
        rel = self.raw["paths"][path_key]
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Charge un fichier YAML en objet :class:`Config`."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=p)
