"""Network mining — embeddings de graphe (DeepWalk) sur les interactions auteur→auteur.

Au lieu de compter des degrés à la main, on *apprend* une représentation vectorielle de chaque
auteur à partir de la structure des réponses : marches aléatoires sur le graphe « qui répond à
qui », puis Word2Vec (skip-gram) sur ces marches (DeepWalk, Perozzi 2014). Deux auteurs qui
évoluent dans les mêmes voisinages conversationnels obtiennent des vecteurs proches.

Mémoire-conscient (WSL ~7 Go) : graphe en listes d'adjacence compactes (numpy CSR-like), marches
générées par lots. Restreint optionnellement aux auteurs actifs (min_degree) pour borner le coût.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_author_graph_embeddings(
    df: pd.DataFrame,
    dim: int = 24,
    num_walks: int = 5,
    walk_length: int = 20,
    window: int = 5,
    min_count_author: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Attend : id, name, author, parent_id. Retourne id + graph_emb_0..graph_emb_{dim-1}.

    Arêtes : (auteur du commentaire) — (auteur du commentaire parent), pour chaque réponse à un
    commentaire (parent t1_). Le graphe est non orienté et non pondéré (les répétitions
    renforcent naturellement via les marches).
    """
    from gensim.models import Word2Vec

    n = len(df)
    author = df["author"].to_numpy()
    author_code, _ = pd.factorize(pd.Series(author), sort=False)

    # position du commentaire-parent -> auteur du parent
    pos_of_name = pd.Series(np.arange(n), index=df["name"].to_numpy())
    pos_of_name = pos_of_name[~pos_of_name.index.duplicated(keep="first")]
    parent_pos = pos_of_name.reindex(df["parent_id"].to_numpy()).to_numpy()
    del pos_of_name
    valid = ~np.isnan(parent_pos)
    src = author_code[valid]
    dst = author_code[parent_pos[valid].astype(np.int64)]
    mask = src != dst  # ignore les self-loops (réponse à soi-même)
    src, dst = src[mask], dst[mask]

    # liste d'adjacence (non orientée) via tri
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    n_nodes = int(author_code.max()) + 1
    deg = np.bincount(u, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(deg)
    neighbors = v  # déjà trié par u

    rng = np.random.default_rng(seed)
    active = np.where(deg >= min_count_author)[0]

    # génération des marches aléatoires (DeepWalk)
    walks = []
    for _ in range(num_walks):
        rng.shuffle(active)
        for start in active:
            walk = [start]
            cur = start
            for _ in range(walk_length - 1):
                s, e = indptr[cur], indptr[cur + 1]
                if e <= s:
                    break
                cur = int(neighbors[rng.integers(s, e)])
                walk.append(cur)
            walks.append([str(x) for x in walk])

    model = Word2Vec(
        walks, vector_size=dim, window=window, min_count=0, sg=1, workers=4,
        epochs=3, seed=seed,
    )

    # vecteur par auteur -> par commentaire
    cols = [f"graph_emb_{i}" for i in range(dim)]
    emb = np.zeros((n_nodes, dim), dtype=np.float32)
    for node in range(n_nodes):
        key = str(node)
        if key in model.wv:
            emb[node] = model.wv[key]
    out = pd.DataFrame(emb[author_code], columns=cols)
    out.insert(0, "id", df["id"].to_numpy())
    return out
