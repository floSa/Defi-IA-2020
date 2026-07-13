"""Feature engineering — deux familles à parts égales (barème 30/30) :

    structural.py   network mining : timing intra-thread, arbre de réponses, auteur, graphe
    stylometric.py  text mining    : longueurs, ponctuation, majuscules, markdown, sentiment
    text.py         text mining    : TF-IDF, embeddings de phrase, topics

Voir docs/plan.md §2 pour la liste détaillée et les précautions anti-fuite (target encoding
calculé dans le fold de CV).
"""
