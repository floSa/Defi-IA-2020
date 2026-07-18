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
| + embeddings **gte-modernbert-base** (128 dims) | 192 | 8,0448 | −0,140 |
| **Deux étages** (classifieur `ups==1` + L1 sur la queue), sur e5 | 128 | **7,9596** | −0,226 |
| **Blend final** (deux étages 0,60 / e5 0,40) | — | **7,9341** | **−0,251 (−3,1 %)** |

Soit **−33,4 %** sur la baseline médiane (11,9074), contre −31,3 % avant cette session.

### Ce que la session a appris
- **Le GPU change la faisabilité, pas seulement la vitesse.** Encoder 4,23 M commentaires prend
  **13 min** avec e5-small (fp16, tri par longueur pour réduire le padding, SVD en streaming) là
  où le kernel Kaggle estimait **47 h en CPU** et se faisait annuler. Facteur ~280.
- **Plus récent et plus gros ≠ meilleur.** `gte-modernbert-base` (2025, 149 M params, variance
  SVD retenue 0,696) fait **moins bien** que `e5-small-v2` (2023, 33 M params, variance 0,583),
  pour **5× le coût d'encodage** (67 min vs 13 min). *Réserve de méthode :* modèle et
  dimensionnalité changent simultanément dans cette comparaison ; le test isolant (tronquer
  modernbert à ses 64 premières composantes SVD, qui sont ordonnées par variance décroissante)
  reste à faire avant de conclure sur le modèle seul.
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

### Prochaines pistes, par rapport coût/bénéfice
1. Tronquer modernbert à 64 dims pour isoler modèle vs dimensionnalité (1 run GBM, ~25 min).
2. e5-small en 128 dims : la SVD à 64 ne retient que 58,3 % de la variance (~35 min).
3. Deux étages appliqué au blend plutôt qu'à un seul modèle.
4. Fine-tuning end-to-end objectif L1 — désormais réaliste, 16 Go de VRAM suffisent pour un
   encodeur *base*.

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
