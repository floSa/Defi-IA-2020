# EDA — constats initiaux (streaming, corpus complet)

Calculé sur les 4 234 970 lignes de `comments_students.csv` via un scan Python
en flux (`scripts/eda_stream.py`, sans dépendance). Ces chiffres pilotent le plan de modélisation.

## Volumétrie
- **Total** : 4 234 970 commentaires.
- **Train** (`ups` renseigné) : 3 218 512 (76,0 %).
- **Test** (`ups = NaN`) : 1 016 458 (24,0 %).

## Cible `ups` (sur le train)
| Stat | Valeur |
|---|---|
| Moyenne | 12,74 |
| Médiane | **1** |
| Min / Max | **−333** / **6761** |
| `ups == 1` | 1 676 837 (**52,1 %**) |
| `ups ≤ 0` | 241 096 (7,49 %) |
| `ups < 0` (downvotés nets) | 87 724 |
| Quantiles (P50/P75/P90/P95/P99) | 1 / 2 / 6 / 15 / 189 |

Distribution grossière : `≤0` 7,5 % · `=1` 52 % · `2–5` 29 % · `6–20` 7 % · `21–100` 2,7 %
· `101–1000` 1,4 % · `>1000` 0,26 %.

**Implications MAE :**
1. La MAE est minimisée par la **médiane** → « prédire 1 partout » est un baseline très solide.
2. La valeur ajoutée d'un modèle vient de la **queue droite** (viral) et des **downvotes**,
   pas du cœur de la distribution.
3. Une transformation `log1p` aide la variance mais **change la géométrie de la MAE** :
   on entraînera plutôt avec des objectifs L1 / Huber / quantile(0.5), et on comparera
   « MAE dans l'espace original » systématiquement.

## ⚠️ Le split train/test est TEMPOREL (constat décisif)
Les plages `created_utc` ne se recouvrent pas :
- **Train** : 2015-05-01 00:00:00 → 2015-05-24 23:59:58 (3 218 512 lignes)
- **Test**  : 2015-05-25 00:00:01 → 2015-05-31 23:59:59 (1 016 458 lignes)

Le fichier est trié : tout le train (24 premiers jours) puis tout le test (7 derniers jours).
**Ce n'est donc pas un split aléatoire mais une prévision temporelle.** Implications :
1. **Validation temporelle obligatoire** : on valide sur les 7 derniers jours du train
   (18–24 mai), modèle appris sur 1–17 mai → même horizon que le test. Un GroupKFold aléatoire
   surestimerait la performance.
2. **Dérive de distribution probable** : les commentaires du test sont plus récents, ont eu moins
   de temps pour accumuler des votes (selon la date du snapshot) → `ups` du test possiblement plus
   bas. À contrôler par **validation adverse**.
3. **Features d'âge** : privilégier l'âge *intra-thread* (âge vs 1er commentaire du thread, portable)
   plutôt que l'âge *absolu* vs snapshot (non stationnaire train→test).
4. **Target encoding auteur** : n'utiliser que le passé (respect de l'ordre temporel) — la
   validation temporelle le garantit naturellement.

## Réseau
- **570 735 auteurs uniques** ; **312 007** lignes avec auteur `[deleted]`/vide.
- **148 848 threads** (`link_id`) ⇒ ~28 commentaires/thread en moyenne.
- **1 seul subreddit** (AskReddit) ⇒ pas de signal inter-subreddit ; tout le network mining
  se joue *à l'intérieur* des threads (arbre de réponses via `parent_id`, timing, auteurs).

## Texte
- Longueur moyenne du `body` : **143 caractères** ; max **10 000** (tronqué à la source).
- 61 corps vides ; markdown brut présent (`**`, `&gt;`, `>`…).

## Pistes de features suggérées par l'EDA
- **Timing intra-thread** (âge du commentaire vs début du thread) : probablement le signal
  structurel le plus fort (visibilité ⇒ upvotes).
- **Rang / position** du commentaire dans le thread, **taille du thread**.
- **Arbre de réponses** : profondeur, nb d'enfants (in-degree), taille de sous-arbre,
  réponse au lien (`t3_`) vs à un commentaire (`t1_`).
- **Auteur** : activité (nb de commentaires), encodage cible historique (avec précaution CV),
  `[deleted]`.
- **Stylométrie** : longueur, ratio majuscules, ponctuation (`!!`, `?`), URLs, markdown,
  émoticônes, sentiment.
- **Sémantique** : embeddings de phrase modernes + topics.
