# Rapport — Défi IA 2020 (revisité 2026) : prédiction des upvotes Reddit

## 1. Contexte
Challenge Kaggle (TSE / Université Paul Sabatier, 2020) : prédire le score `ups` d'un commentaire
du subreddit **AskReddit** (mai 2015) à partir de son texte et de méta-données, en combinant
**text mining** et **network mining**. Métrique officielle : **MAE** (à minimiser). Scores
historiques de la compétition : ~8 à 11.

## 2. Données & EDA (constats décisifs)
- **4 234 970 commentaires** : 3 218 512 train / 1 016 458 test (24 %).
- **Split TEMPOREL** (découvert à l'EDA, non aléatoire) : train = 1→24 mai, test = 25→31 mai,
  sans recouvrement. → C'est une **prévision**. Validation = **holdout temporel** sur les 7
  derniers jours du train (même horizon que le test) ; un K-fold aléatoire surestimerait la perf.
- **Cible ultra-asymétrique** : médiane = 1, moyenne 12,7, min −333, max 6761, **52 % des `ups`
  valent exactement 1**. Conséquence majeure : la MAE étant minimisée par la médiane, le cœur de
  la distribution ne pèse presque rien ; les ~11,9 points d'erreur du baseline viennent de la
  **queue virale**. **Le problème est essentiellement une régression de queue** — tout le gain
  consiste à prédire *qui* devient viral et *de combien*.

## 3. Méthodologie
- **Validation** : holdout temporel 7 jours (`src/defia/evaluation/cv.py`). MAE toujours mesurée
  en espace original.
- **Features réseau (network mining)** : âge intra-thread, rang/percentile temporel dans le fil,
  taille du thread & nb d'auteurs, arbre de réponses via `parent_id` (profondeur, nb d'enfants,
  nb de frères, rang parmi les frères, réponse au lien vs à un commentaire), activité auteur,
  **target encoding auteur temporellement propre** (moyenne bayésienne du karma passé, expanding
  leave-one-out — sans fuite).
- **Features texte (text mining)** : stylométrie (longueurs, ponctuation, majuscules, markdown,
  émoticônes), sentiment **VADER**, et **TF-IDF** (HashingVectorizer word 1–2-grams + TruncatedSVD
  32 dimensions).
- **Modèle** : **LightGBM**, objectif `mae` (L1) — le mieux adapté à la queue lourde (bat Huber).
- **Blending** : poids optimisant directement la MAE sur le holdout (`src/defia/models/blend.py`).

## 4. Résultats (MAE, holdout temporel)
| Modèle | MAE | vs baseline |
|---|---|---|
| Baseline — médiane (1) | 11.907 | — |
| Médiane par auteur (naïf) | 13.253 | pire (encodage naïf nuisible) |
| GBM `huber` | 10.459 | +12.2 % |
| GBM `mae` — réseau + stylométrie | 8.395 | +29.5 % |
| GBM `mae` + target encoding auteur | 8.356 | +29.8 % |
| GBM `mae` + TF-IDF | 8.284 | +30.4 % |
| **GBM `mae` — 64 features en full** (contexte + dynamique auteur + parent-enc + interactions + TF-IDF) | **8.178** | **+31.3 %** (meilleur modèle) |
| **Blend final** (full 0.73 + tfidf 0.27) | **8.162** | **+31.5 %** |

Le modèle « 64 features en full » est entraîné **sans contrainte mémoire** (kernel Kaggle CPU
30 Go RAM, 0 GPU), sur les 3,2 M lignes et les blocs de features avancées (features de contexte
intra-fil, dynamique auteur enrichie, réputation auteur du parent, interactions, TF-IDF). Le laptop
(7,4 Go RAM) ne peut pas charger l'ensemble : l'offload CPU Kaggle est ce qui débloque le full.
Le blend final absorbe entièrement le champion (poids 0) : il ne reste que le full et le TF-IDF.

**Soumission finale** : `submissions/submission_final3.csv` (1 016 458 lignes, format `id,predicted`).

## 5. Enseignements
- **Le network mining porte l'essentiel du signal** : sur les features les plus importantes,
  les structurelles (taille du thread, position/timing intra-thread, arbre de réponses)
  dominent très largement les features texte. C'est cohérent : sur AskReddit, la **visibilité
  dans le fil** prédit la viralité mieux que le style.
- **Le TF-IDF apporte un complément texte réel** : +0,07 de MAE (8,356 → 8,284), le signal
  lexical capte ce que la stylométrie seule manquait.
- **Les features avancées + le full data font le gros du dernier gain** : contexte intra-fil,
  dynamique auteur enrichie, réputation de l'auteur du parent et interactions, entraînés sur les
  3,2 M lignes complètes, font passer de 8,284 à **8,178** (−0,106). L'essentiel de ce gain vient
  de l'entraînement sur toutes les lignes (le laptop devait sous-échantillonner) autant que des
  nouvelles features.
- **MAE = régression de queue** : l'objectif L1 direct bat Huber et les transformations log.
- **Rigueur anti-fuite** : la feature `parent_ups` donnait un MAE trompeur de 7,84 mais n'est
  disponible que sur 1,5 % du test réel (vs 59 % du holdout) — écartée. Seule la réputation
  agrégée de l'auteur du parent (disponible partout) est conservée.

## 6. Session GPU locale (RTX 4060 Ti 16 Go) — passage sous 8,0

Le pari du § précédent (« les prédictions texte profondes devraient faire passer sous 8,0 »)
est **vérifié**. Toutes les MAE ci-dessous sont mesurées sur le même holdout temporel 7 jours.

| Modèle | Features | MAE | vs référence |
|---|---|---|---|
| Champion CPU reproduit en local | 64 | 8,1851 | référence |
| + embeddings **e5-small-v2** (64 dims) | 128 | **7,9968** | −0,188 |
| + embeddings **gte-modernbert-base** (64 dims, convergé) | 128 | 7,9996 | −0,186 |
| **Deux étages** (classifieur `ups==1` + L1 sur la queue), sur e5 | 128 | **7,9596** | −0,226 |
| **Blend** (deux étages 0,60 / e5 0,40) | — | 7,9341 | −0,251 |
| **Blend + règle deux étages par-dessus** | — | **7,9276** | **−0,258 (−3,1 %)** |

Soit **−33,4 %** sur la baseline médiane (11,9074), contre −31,3 % avant cette session.
Soumission finale : `submissions/submission_final_blend2s.csv`.

### Ablation des embeddings — tous les points convergés
| Modèle | Dims | MAE | best_iter | Plafond |
|---|---|---|---|---|
| e5-small-v2 | 16 | 8,0264 | 4 286 | 8 000 |
| **e5-small-v2** | **32** | **7,9973** | 5 653 | 6 000 |
| **e5-small-v2** | **64** | **7,9968** | 2 784 | 3 000 |
| e5-small-v2 | 128 | 8,0325 | 3 938 | 6 000 |
| **gte-modernbert-base** | **64** | **7,9996** | **10 176** | 12 000 |
| gte-modernbert-base | 128 | 8,0067 | 7 459 | 12 000 |

**Courbe de dimensionnalité en U**, plateau optimal entre 32 et 64 : à 16 dims on a jeté du
signal utile, à 128 on donne au GBM trop d'occasions d'ajuster du bruit. La variance retenue par
la SVD n'est **pas** le bon critère de choix (0,736 à 128 dims fait moins bien que 0,583 à 64).

**Les deux encodeurs sont équivalents en précision** : 7,9996 contre 7,9968, soit 0,0028 —
en dessous du bruit de LightGBM (deux runs identiques sur machines différentes donnaient déjà
0,007 d'écart). Le choix se fait donc sur le **coût**, où e5 domine nettement :

| | e5-small-v2 | gte-modernbert-base |
|---|---|---|
| Encodage de 4,23 M textes | **13 min** | 67 min |
| Arbres nécessaires pour converger | **2 784** | 10 176 |
| MAE | 7,9968 | 7,9996 |

→ **e5-small-v2 reste le bon choix, pour son coût, pas pour sa précision.**

### Le piège de mesure qui a faussé cette section pendant trois heures
Ce rapport a d'abord affirmé — et un commit a publié — que *« gte-modernbert-base (2025) perd
contre e5-small-v2 (2023) »*, avec un écart de 0,048. **C'était faux.** Les runs modernbert
s'arrêtaient au plafond d'arbres sans que l'early-stopping ait jamais pu se déclencher : ils
progressaient encore. La MAE mesurait le **budget d'entraînement**, pas la qualité des features.

Correction du chiffre au fil des budgets, pour modernbert-64 :
`8,0455 (3 000) → 8,0195 (6 000) → 7,9996 (12 000, convergé)`.

L'erreur valait 0,046 — **dix fois** l'écart réel entre les deux modèles, et sept fois le gain
que l'ablation cherchait à détecter. Un garde-fou est désormais dans `cli.py` : il alerte dès que
`best_iter + patience > plafond`. C'est lui qui a détecté deux des runs fautifs, après coup.

**Règle à retenir** : comparer des configurations à nombre de features différent exige de
vérifier que *chacune* a convergé. Plus de features (ou des features plus difficiles à exploiter)
demande plus d'arbres ; à budget fixe, on classe les budgets et non les modèles.

### La règle « réponds 1 » appliquée au blend
Gain réel mais faible (−0,0065, confirmé à −0,0069 sur la moitié non vue) : le blend contient
déjà le modèle deux étages à 60 %, la règle y était donc à moitié appliquée. Le seuil retenu
(0,45) force **68 % des prédictions test à 1** alors que 52 % des commentaires valent réellement
1 — sur-prédire la médiane est optimal en MAE, mais ce réglage serait mauvais pour toute
métrique sensible à la moyenne (RMSE).

### Ce que la session a appris
- **Le GPU change la faisabilité, pas seulement la vitesse.** Encoder 4,23 M commentaires prend
  **13 min** avec e5-small (fp16, tri par longueur pour réduire le padding, SVD en streaming) là
  où le kernel Kaggle estimait **47 h en CPU** et se faisait annuler. Facteur ~280.
- **Plus récent et plus gros ≠ plus précis, mais surtout : beaucoup plus cher.**
  `gte-modernbert-base` (2025, 149 M params) et `e5-small-v2` (2023, 33 M) atteignent la **même
  MAE** (7,9996 vs 7,9968, écart sous le bruit). Le récent coûte 5× plus à encoder et 3,7× plus
  d'arbres à exploiter — d'où le choix d'e5. *Cette conclusion a d'abord été fausse dans le sens
  inverse* (« modernbert perd de 0,048 ») faute de budget d'arbres suffisant ; voir la section
  sur le piège de mesure, qui est probablement l'enseignement le plus transférable de la session.
- **Le signal texte est diffus.** Aucun `emb_*` n'entre dans le top-15 des features par gain —
  les structurelles dominent toujours — et pourtant les 64 dimensions valent ensemble 0,19 point
  de MAE. Conséquence pratique : ne pas élaguer ces features sur un critère de gain individuel.
- **La forme de la cible se traite explicitement.** 52 % des `ups` valent 1 ; séparer
  « est-ce un 1 ? » (AUC 0,793) de « quelle valeur en queue ? » gagne 0,037. Le seuil de
  combinaison est réglé sur la première moitié du holdout et **mesuré sur la seconde, jamais vue
  par le réglage** : le gain y est même plus large (−0,055), ce n'est donc pas un artefact.
- **Le blend écarte tout le passé.** Poids nul pour les trois modèles antérieurs
  (`gbm_champion`, `gbm_kaggle`, `gbm_tfidf`) : ils n'apportent plus rien face aux deux modèles
  à embeddings.

### Deux bugs de fond corrigés au passage
- **Cap de ré-entraînement à 2 M lignes en dur** (hérité du laptop 7,4 Go) : il amputait le
  modèle final de 38 % du train sur une machine 66 Go. Désormais calculé depuis la RAM
  disponible. Impact mesuré : **47 % des prédictions test bougent de plus de 0,1**.
- **`UnicodeEncodeError` sur `★`** dans la sortie du blend : la console Windows est en cp1252,
  le run entier plantait *après* que le blend avait abouti, résultat perdu.

### Prochaines pistes
Les pistes 1 à 3 de la liste initiale ont été exécutées (résultats ci-dessus) : les deux
premières sont des **résultats négatifs**, la troisième un gain marginal. On est entré dans les
rendements décroissants côté assemblage — les trois derniers gains valent 0,037, puis 0,026,
puis 0,0065.

Ce qui reste porte donc sur le **signal**, pas sur la combinaison :
1. **Fine-tuning end-to-end objectif L1** : entraîner l'encodeur *sur la tâche* au lieu
   d'utiliser des embeddings figés. C'est le seul levier qui peut encore apporter du signal
   texte nouveau. Réaliste ici (16 Go de VRAM suffisent pour un encodeur *base*).
2. **Dimensionnalité en dessous de 64** : puisque 128 dégrade et 64 gagne, l'optimum n'a pas été
   encadré par le bas — 32 dims n'a jamais été testé.
3. **Features de graphe** sur l'arbre de réponses (node2vec avait échoué ; un GNN reste ouvert).

### Impasses Kaggle (GPU) rencontrées
- GPU **P100 (sm_60)** attribué : incompatible avec le PyTorch pré-installé de Kaggle (sm_70+).
  Contournement (réinstaller torch) casse `transformers` (versions désalignées) ; réinstaller
  torch+transformers ensemble bute ensuite sur le **quota GPU Kaggle** (2 sessions max).
- Embeddings en CPU sur Kaggle : trop lents (> 7 h) → annulés.
- → Les briques GPU tourneront proprement sur une machine à GPU récent (torch/transformers natifs).

## 7. Reproductibilité
```bash
make data              # CSV -> parquet (split temporel)
make features          # features réseau + stylométrie + VADER
make author-encoding   # target encoding auteur temporel
make tfidf-features    # TF-IDF hashing + SVD
make train-gbm         # LightGBM (objectif MAE) -> OOF + soumission
make blend             # blend des OOF -> soumission finale
```
Pipeline agnostique à la machine, piloté par `configs/default.yaml`. Détails d'implémentation et
journal complet dans `docs/guide.md`.
