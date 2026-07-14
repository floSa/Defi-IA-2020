"""Feature engineering — interactions non linéaires et encodage temporel cyclique.

Les GBM capturent des interactions par les arbres, mais expliciter les produits les plus
plausibles métier aide (moins d'arbres, signal plus net) :
  * réputation auteur × moment d'arrivée (un bon auteur qui poste tôt),
  * vélocité du thread × profondeur (fil qui s'emballe vs enfoui),
  * propension au downvote de l'auteur × longueur (auteur clivant qui écrit long).
Encodage cyclique heure/jour (sin/cos) : 23h et 0h sont proches, un entier ne le capture pas.

Lit les parquets de features déjà produits ; ne recalcule rien de lourd.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_interactions(base: pd.DataFrame, context: pd.DataFrame, author_dyn: pd.DataFrame) -> pd.DataFrame:
    """base = train_features (id, hour, dow, n_chars), context, author_dyn — tous alignables par id."""
    df = base[["id", "hour", "dow", "n_chars", "depth"]].merge(
        context[["id", "arrival_frac", "thread_velocity"]], on="id", how="left").merge(
        author_dyn[["id", "author_dyn_mean", "author_dyn_down_frac"]], on="id", how="left")

    out = pd.DataFrame({"id": df["id"].to_numpy()})
    out["x_authormean_arrival"] = (df["author_dyn_mean"] * df["arrival_frac"]).astype(np.float32)
    out["x_velocity_depth"] = (df["thread_velocity"] * df["depth"]).astype(np.float32)
    out["x_downfrac_len"] = (df["author_dyn_down_frac"] * np.log1p(df["n_chars"])).astype(np.float32)

    h = df["hour"].to_numpy(dtype=float)
    d = df["dow"].to_numpy(dtype=float)
    out["hour_sin"] = np.sin(2 * np.pi * h / 24).astype(np.float32)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24).astype(np.float32)
    out["dow_sin"] = np.sin(2 * np.pi * d / 7).astype(np.float32)
    out["dow_cos"] = np.cos(2 * np.pi * d / 7).astype(np.float32)
    return out
