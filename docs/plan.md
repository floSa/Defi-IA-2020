# Plan de travail — Défi IA 2020 revisité (état de l'art 2026)

> Document à **valider** avant d'écrire le code de modélisation. Objectif : reconstruire une
> solution propre, reproductible et **meilleure que l'état de l'art de 2020**, en couvrant
> rigoureusement les deux volets exigés (text mining + network mining).

## 0. Objectif & garde-fous
- **Métrique** : MAE (à minimiser). Baseline de référence = « prédire la médiane (1) ».
- **Cible secondaire** : battre le classement d'origine (à préciser : quel était ton rang /
  score MAE ? ça fixe la barre à viser).
- **Validation** : `GroupKFold` par `link_id` (thread) pour éviter la fuite intra-thread ;
  toutes les features à risque de fuite (target encoding auteur/thread) calculées **dans le
  fold**. On rapporte toujours la MAE OOF dans l'espace original.

## 1. Data layer (`src/defia/data`)
- Chargement typé + robuste (dtypes, gestion `NaN` cible, échappements HTML `&lt; &gt; &amp;`).
- Découpage train/test dérivé de `ups` (NaN = test), persistance en **parquet** (`data/interim`).
- Nettoyage texte réversible (on garde le brut ET une version nettoyée : le markdown est un
  signal, pas seulement du bruit).

## 2. Feature engineering — deux familles à parts égales

### 2a. Network / structural mining (30 pts du barème)
- **Timing intra-thread** : `created_utc` du commentaire − `created_utc` min du thread
  (âge relatif). Rang temporel du commentaire dans le thread, percentile temporel.
  *(Hypothèse EDA : signal structurel le plus fort — visibilité ⇒ upvotes.)*
- **Taille & popularité du thread** : nb de commentaires par `link_id`, nb d'auteurs distincts.
- **Arbre de réponses** (via `parent_id`) : profondeur du commentaire, réponse au lien
  (`t3_`) vs à un commentaire (`t1_`), nombre d'enfants (in-degree), taille de sous-arbre,
  nombre de frères/sœurs, rang parmi les frères. Construit par thread (graphe orienté).
- **Auteur** : activité (nb commentaires dans le corpus), `is_deleted`, encodage cible
  historique de l'auteur (moyenne/médiane `ups` — **hold-out par fold**), part de commentaires
  downvotés de l'auteur.
- **Graphe d'interactions** (bonus) : graphe auteur→auteur (qui répond à qui) ; degré,
  PageRank/betweenness approximés au niveau thread.

### 2b. Text mining (30 pts du barème)
- **Stylométrie** : longueurs (car., mots, phrases), longueur moyenne des mots, ratio
  majuscules, comptages de ponctuation (`!`, `?`, `!!`, `...`), présence d'URL, de markdown
  (`**`, citations `>`), d'émoticônes/emoji, question vs affirmation, richesse lexicale
  (type-token ratio), profanité, lisibilité (Flesch).
- **Sentiment** : VADER (compound, pos/neg/neu) — rapide, CPU.
- **Lexical classique** : TF-IDF word + char n-grams (hashing pour tenir en RAM) → utilisé
  par un modèle linéaire (feature/baseline) et/ou réduit (SVD) pour le GBM.
- **Topical** : topics par clustering d'embeddings (ou BERTopic) — le commentaire aborde-t-il
  un sujet « porteur » ; distance au thread.

### 2c. Sémantique profonde (étapes C/D — GPU)
- **Embeddings de phrase modernes** (SOTA 2024–2026) : `e5`/`bge`/`gte`-base ou `ModernBERT`
  → vecteurs 384–768 dim, injectés dans le GBM (late fusion tabulaire+texte).
- **Fine-tuning encodeur** : `DistilBERT` / `ModernBERT-base` / `DeBERTa-v3-small` sur
  `body → ups` (tête de régression, loss L1/Huber). Produit des **prédictions OOF** pour le
  blending. *(Voir `compute_strategy.md` pour où l'entraîner : desktop 4060 Ti / Kaggle / Colab.)*

## 3. Modèles
| # | Modèle | Rôle | Où |
|---|---|---|---|
| B0 | Constante = médiane (1) | Plancher MAE | local |
| B1 | Médianes par groupe (thread / heure / auteur) | Baseline malin | local |
| M1 | **LightGBM** (objectif L1/Huber/quantile-0.5) sur toutes features tabulaires | Cheval de bataille | local CPU |
| M2 | CatBoost (gère la haute cardinalité auteur/link) | Alternative GBM | local CPU |
| M3 | Linéaire sur TF-IDF (ElasticNet / SGD) | Vue texte pure | local CPU |
| M4 | **Transformer fine-tuné** (body → ups) | Vue sémantique | GPU distant |
| M5 | GBM + embeddings de phrase (late fusion) | Fusion | local CPU (+GPU pour embeddings) |
| **ENS** | **Blending OOF** (ridge/LGBM meta, poids optimisés MAE) | Modèle final | local |

### Traitement de l'asymétrie (spécifique MAE)
- Objectifs L1 / Huber / quantile(0.5) plutôt que MSE.
- Option **deux étages** : (a) classer `downvoté / normal / viral`, (b) régresser dans chaque
  régime — souvent plus robuste sur queues lourdes que la régression directe.
- Comparaison systématique `log1p` vs espace direct, MAE toujours mesurée en espace original.

## 4. Ce qui dépasse l'état de l'art 2020 (l'ask « état de l'art à jour »)
- Encodeurs **ModernBERT / DeBERTa-v3 / e5-bge-gte** (2024+) vs BERT-base de 2020.
- **CatBoost / LightGBM** avec objectif L1 natif et meilleur handling catégoriel.
- **Late fusion** texte-embeddings + tabulaire (approche « TabText ») avec tête MLP légère.
- Modélisation **deux étages** de la queue + calibration MAE.
- Features **LLM zero-shot** optionnelles (qualité/topic via petit LLM local) — dans le respect
  strict de la règle « pas de re-téléchargement du dataset complet ».

## 5. Évaluation & reproductibilité
- CV `GroupKFold(link_id)` ; tracking des MAE OOF par modèle et de l'ensemble.
- Chaque étape orchestrée par le `Makefile` (`make data`, `make features`, `make train-*`,
  `make blend`, `make submit`) et paramétrée par `configs/*.yaml`.
- Artefacts versionnés en parquet/json ; submission au format `id,predicted`.
- Rapport final `reports/` : contribution de chaque feature/modèle, courbes, ablations.

## 6. Séquencement proposé
1. **A** — data layer + baselines B0/B1 (chiffre le plancher MAE réel). *(local)*
2. **B** — features réseau + stylométriques + M1 LightGBM. *(local)* → 1re soumission.
3. **C** — embeddings de phrase + M5 late fusion. *(embeddings GPU, GBM local)*
4. **D** — fine-tuning transformer M4. *(GPU distant)*
5. **E** — blending ENS + ablations + rapport + soumission finale. *(local)*

## Questions ouvertes à trancher avec toi
1. **Route GPU** pour C/D : desktop 4060 Ti / Kaggle Kernels / Colab+Drive ? (cf. `compute_strategy.md`)
2. **Barre à viser** : quel était ton score MAE / classement d'origine ?
3. **Périmètre** : on vise le pipeline complet (A→E) ou on cadence par jalons validés ?
4. **Soumissions Kaggle** : la compétition est-elle encore ouverte en late-submission, ou on
   se contente d'une évaluation MAE OOF hors-ligne comme juge de paix ?
