"""Network mining avancé — features de CONTEXTE (parent, vélocité de thread, dynamique d'auteur).

Va au-delà des features structurelles de base : on caractérise *ce à quoi le commentaire répond*
et *le moment de la conversation où il arrive*. Intuition : une répartie rapide sous un
commentaire-carrefour, tôt dans un fil qui s'emballe, capte la visibilité — donc les upvotes.

Calculé sur l'union train+test (transductif, sans label). Vectorisé numpy pour la volumétrie.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from defia.features.structural import _grouped_temporal_rank


def build_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attend : id, created_utc, link_id, name, author, parent_id. Retourne id + features de contexte."""
    n = len(df)
    out = pd.DataFrame(index=np.arange(n))
    out["id"] = df["id"].to_numpy()
    created = df["created_utc"].to_numpy(dtype=np.int64)

    link_code = pd.factorize(df["link_id"], sort=False)[0].astype(np.int64)
    author_code = pd.factorize(df["author"], sort=False)[0].astype(np.int64)

    # position du commentaire-parent (ou -1 si réponse au lien / parent absent)
    pos_of_name = pd.Series(np.arange(n), index=df["name"].to_numpy())
    pos_of_name = pos_of_name[~pos_of_name.index.duplicated(keep="first")]
    parent_pos = pos_of_name.reindex(df["parent_id"].to_numpy()).to_numpy()
    del pos_of_name
    parent_pos = np.where(np.isnan(parent_pos), -1, parent_pos).astype(np.int64)
    has_parent = parent_pos >= 0
    pp = np.where(has_parent, parent_pos, 0)  # index sûr

    # rang temporel & profondeur (pour caractériser le parent)
    rank = _grouped_temporal_rank(link_code, created)
    depth = np.zeros(n, dtype=np.int32)
    cur = parent_pos.copy()
    for _ in range(200):
        m = cur >= 0
        if not m.any():
            break
        depth[m] += 1
        cur[m] = parent_pos[cur[m]]
    n_children = np.bincount(parent_pos[has_parent], minlength=n).astype(np.int32)

    # --- Contexte du parent ---
    out["time_gap_to_parent"] = np.where(has_parent, np.log1p(np.maximum(created - created[pp], 0)), -1.0).astype(np.float32)
    out["parent_rank_in_thread"] = np.where(has_parent, rank[pp], -1).astype(np.int32)
    out["parent_n_children"] = np.where(has_parent, n_children[pp], 0).astype(np.int32)  # le parent est-il un carrefour ?
    out["parent_depth"] = np.where(has_parent, depth[pp], -1).astype(np.int32)
    out["is_reply_to_own"] = (has_parent & (author_code == author_code[pp])).astype(bool)

    # --- Vélocité & cycle de vie du thread ---
    g = pd.DataFrame({"l": link_code, "t": created})
    tmin = g.groupby("l")["t"].transform("min").to_numpy()
    tmax = g.groupby("l")["t"].transform("max").to_numpy()
    size = np.bincount(link_code, minlength=1)[link_code].astype(np.float64)
    dur = np.maximum(tmax - tmin, 1)
    out["thread_duration_log"] = np.log1p(dur).astype(np.float32)
    out["thread_velocity"] = (size / dur).astype(np.float32)          # commentaires / seconde
    out["arrival_frac"] = ((created - tmin) / dur).astype(np.float32)  # position dans la vie du fil [0,1]

    # --- Dynamique de l'auteur DANS le thread (combien de commentaires déjà postés avant) ---
    at_code = link_code * (author_code.max() + 1) + author_code  # clé (thread, auteur)
    out["author_prior_in_thread"] = _grouped_temporal_rank(at_code, created).astype(np.int32)

    return out
