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

## 6. Ce qui reste (état de l'art à pousser)
- **Fine-tuning d'un encodeur** (DistilBERT / ModernBERT) et **embeddings de phrase** (e5) sur
  `body` : code prêt (`scripts/kaggle/`, `src/defia/models/transformer.py`), en attente d'un GPU
  disponible. Impasses rencontrées côté Kaggle documentées ci-dessous.
- Intégration de ces prédictions texte profondes au blend devrait faire passer sous 8,0.

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
