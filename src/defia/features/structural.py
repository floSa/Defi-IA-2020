"""Network mining — features structurelles (volet 30 pts du barème).

Calculées sur l'**union train+test** (transductif, sans utiliser la cible `ups`) pour un contexte
de thread correct : un thread (`link_id`) peut chevaucher la frontière temporelle train/test.

Contrainte mémoire (WSL ~7 Go, 4,2 M lignes) : on **factorise immédiatement les colonnes texte
en codes entiers int32** et on ne conserve que du numérique. `link_code` / `author_code` servent
en aval (CV par thread, target encoding auteur) ; les strings `link_id`/`author` ne sont pas
gardées.

Familles (cf. docs/plan.md §2a) : timing intra-thread, taille de thread, arbre de réponses
(`parent_id`/`name`), auteur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MAX_DEPTH_ITERS = 200  # garde-fou pour la remontée d'arbre


def _grouped_temporal_rank(codes: np.ndarray, created: np.ndarray) -> np.ndarray:
    """Rang temporel (0-indexé) de chaque ligne au sein de son groupe (codes entiers)."""
    n = len(codes)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((created, codes))
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    sorted_codes = codes[order]
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = sorted_codes[1:] != sorted_codes[:-1]
    grp_start = np.maximum.accumulate(np.where(change, np.arange(n), 0))
    return (np.arange(n) - grp_start)[inv]


def build_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features structurelles pour l'union des commentaires (numérique, mémoire-léger).

    Attend : id, created_utc, link_id, name, author, parent_id.
    Retourne un DataFrame aligné sur l'ordre de ``df`` (colonnes numériques + 'id').
    """
    n = len(df)
    out = pd.DataFrame(index=np.arange(n))
    out["id"] = df["id"].to_numpy()
    created = df["created_utc"].to_numpy(dtype=np.int64)

    # --- Codes entiers (libère les strings au plus tôt) ---
    link_code = pd.factorize(df["link_id"], sort=False)[0].astype(np.int32)
    author_is_deleted = pd.Series(df["author"].to_numpy()).isin(["[deleted]"]).fillna(True).to_numpy()
    author_code = pd.factorize(df["author"], sort=False)[0].astype(np.int32)
    parent_id = df["parent_id"].to_numpy()
    is_reply_to_link = pd.Series(parent_id).str.startswith("t3_").fillna(False).to_numpy()

    # parent_pos : ligne du commentaire-parent (ou -1). Série string-indexée transitoire.
    pos_of_name = pd.Series(np.arange(n), index=df["name"].to_numpy())
    pos_of_name = pos_of_name[~pos_of_name.index.duplicated(keep="first")]
    parent_pos = pos_of_name.reindex(parent_id).to_numpy()
    del pos_of_name
    parent_pos = np.where(np.isnan(parent_pos), -1, parent_pos).astype(np.int64)

    # --- Thread-level (via codes) ---
    thread_size = np.bincount(link_code, minlength=1)[link_code].astype(np.int32)
    thread_start = pd.Series(created).groupby(link_code, sort=False).transform("min").to_numpy()
    tmp = pd.DataFrame({"l": link_code, "a": author_code})
    thread_n_authors = tmp.groupby("l", sort=False)["a"].transform("nunique").to_numpy().astype(np.int32)
    del tmp
    rank = _grouped_temporal_rank(link_code, created)

    out["thread_size"] = thread_size
    out["thread_n_authors"] = thread_n_authors
    out["age_in_thread"] = (created - thread_start).astype(np.int32)
    out["rank_in_thread"] = rank.astype(np.int32)
    out["pct_in_thread"] = (rank / np.maximum(thread_size - 1, 1)).astype(np.float32)

    # --- Arbre de réponses ---
    out["is_reply_to_link"] = is_reply_to_link
    valid = parent_pos[parent_pos >= 0]
    n_children = np.bincount(valid, minlength=n).astype(np.int32)
    out["n_children"] = n_children
    out["has_children"] = n_children > 0

    depth = np.zeros(n, dtype=np.int32)
    cur = parent_pos.copy()
    for _ in range(_MAX_DEPTH_ITERS):
        mask = cur >= 0
        if not mask.any():
            break
        depth[mask] += 1
        cur[mask] = parent_pos[cur[mask]]
    out["depth"] = depth

    parent_code = pd.factorize(df["parent_id"], sort=False)[0].astype(np.int32)
    out["n_siblings"] = np.bincount(parent_code, minlength=1)[parent_code].astype(np.int32)
    out["sibling_rank"] = _grouped_temporal_rank(parent_code, created).astype(np.int32)

    # --- Auteur ---
    out["author_n_comments"] = np.bincount(author_code, minlength=1)[author_code].astype(np.int32)
    out["author_is_deleted"] = author_is_deleted

    # --- Temps calendaire ---
    out["hour"] = ((created // 3600) % 24).astype(np.int8)
    out["dow"] = (((created // 86400) + 4) % 7).astype(np.int8)  # epoch day 0 = jeudi

    # --- Méta (codes entiers, pour CV/target-encoding en aval) ---
    out["link_code"] = link_code
    out["author_code"] = author_code
    return out
