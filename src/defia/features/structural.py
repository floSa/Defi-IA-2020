"""Network mining — features structurelles (volet 30 pts du barème).

Familles (cf. docs/plan.md §2a) :
  * Timing intra-thread : âge du commentaire vs 1er commentaire du thread, rang/percentile
    temporel. (Hypothèse EDA : signal structurel le plus fort.)
  * Thread : taille (nb commentaires par link_id), nb d'auteurs distincts.
  * Arbre de réponses (parent_id) : profondeur, réponse au lien (t3_) vs commentaire (t1_),
    nb d'enfants (in-degree), taille de sous-arbre, nb de frères, rang parmi les frères.
  * Auteur : activité, is_deleted, target encoding historique (hold-out par fold), part downvotée.
  * Graphe d'interactions auteur->auteur : degré, PageRank approximé (networkx).

TODO (post-validation) : implémenter les extracteurs ci-dessous.
"""
from __future__ import annotations


def add_timing_features(df):
    raise NotImplementedError("À implémenter après validation du plan.")


def add_thread_features(df):
    raise NotImplementedError("À implémenter après validation du plan.")


def add_reply_tree_features(df):
    raise NotImplementedError("À implémenter après validation du plan.")


def add_author_features(df, target_col: str = "ups"):
    raise NotImplementedError("À implémenter après validation du plan.")
