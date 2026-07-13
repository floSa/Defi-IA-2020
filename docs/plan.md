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

## 4. Stack « état de l'art 2026 » — ce qui nous fait dépasser 2020

Une solution 2020 typique = TF-IDF + features à la main + XGBoost, éventuellement BERT-base.
Voici les briques **récentes** qu'on mobilise, avec le niveau d'engagement.

### 4.1 Encodeurs de texte (2024–2026) — **par défaut**
- **ModernBERT-base** (déc. 2024) pour le fine-tuning `body → ups` : plus rapide, contexte long,
  nettement au-dessus de BERT/RoBERTa 2020. Drop-in idéal.
- **Embeddings de phrase modernes** (top MTEB 2025) pour injection GBM : `bge`/`gte`/`nomic-embed`
  *base*. On exploite les **embeddings Matryoshka** (Nomic/bge) → on tronque à 128–256 dims pour
  un coût GBM faible sans ré-entraîner.

### 4.2 Features dérivées d'un LLM (le vrai saut vs 2020) — **par défaut, léger**
Un commentaire AskReddit devient viral par l'**humour / la punchline / la relatabilité**, pas par
le TF-IDF. On extrait des signaux qu'un LLM sait juger et pas un sac-de-mots :
- **Scoring zéro-shot** via un petit LLM instruct local (Qwen2.5-0.5B/1.5B, Llama-3.2) :
  humour, caractère polémique, effort/qualité, ton, présence de storytelling. Sorties
  structurées → colonnes GBM.
- **Têtes de classification spécialisées** prêtes à l'emploi : toxicité (Detoxify), émotions
  (GoEmotions), sarcasme. Rapide, GPU-léger.
> Respecte strictement la règle « pas de re-téléchargement du dataset Reddit complet » : on
> n'utilise que des modèles pré-entraînés publics, pas un dump de labels.

### 4.3 Network mining moderne — **par défaut (hand-features) + optionnel (embeddings de graphe)**
- Hand-features structurelles (cf. §2a) : socle robuste.
- **Embeddings de graphe** (upgrade 2020→2026) : `node2vec`/GraphSAGE sur le **graphe
  d'interactions auteur→auteur** et la structure de l'arbre de réponses → vecteurs d'auteur/thread
  injectés dans le GBM. Remplace les features de centralité faites main par du représentationnel.

### 4.4 Modèle tabulaire & gestion de la queue lourde — **par défaut**
- **LightGBM/CatBoost** objectif L1/quantile natif ; **Optuna** (TPE multivarié + Hderband) pour
  l'HPO.
- **Modèle en deux parties (hurdle)** adapté à la distribution : (a) P(downvoté / =1 / >1),
  (b) régression de la queue conditionnellement — puis recomposition. Naturel ici vu les 52 % à 1.
- **Régression distributionnelle / quantile(0.5)** : on prédit la médiane conditionnelle, exactement
  ce que la MAE récompense (option LightGBM quantile ou têtes multi-quantiles).

### 4.5 Fusion multimodale — **par défaut (late) / optionnel (jointe)**
- **Late fusion** (embeddings + LLM-features + tabulaire → GBM) : robuste, on commence par là.
- Optionnel : **tête de fusion jointe** (MLP sur `concat[CLS transformer, features tabulaires]`)
  entraînée bout-en-bout — approche « TabText » 2023+.

### 4.6 Robustesse & validation — **par défaut**
- **Validation adverse** train-vs-test (le test = 24 %, possible dérive temporelle sur mai 2015).
- Blending final optimisé MAE (ridge/NNLS/LGBM) sur les OOF, + ablations par famille.
- (Optionnel) **prédiction conforme** pour des intervalles calibrés — bonus analytique, pas requis
  par la MAE ponctuelle.

### Ordre de priorité (rapport gain/effort décroissant)
1. GBM + hand-features réseau/texte + objectif MAE/quantile (socle, étape B).
2. Embeddings modernes en late fusion (étape C).
3. LLM zero-shot features (étape C bis — fort différenciateur vs 2020).
4. Fine-tuning ModernBERT (étape D).
5. Embeddings de graphe + hurdle model + fusion jointe (raffinements, étape E).

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
