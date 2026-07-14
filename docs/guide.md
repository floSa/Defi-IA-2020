# Guide de développement & journal de bord — Défi IA 2020

Document vivant : **comment travailler sur le repo** (haut) + **journal de bord chronologique**
des décisions et résultats (bas). Mis à jour et commité à chaque jalon.

---

## 1. Comment lancer le projet

### Environnement
Le projet vit dans WSL Ubuntu (`/home/florian/mes_projets/Defi-IA 2020`). Environnement Python
géré par **`uv`** (venv `.venv`, gitignoré).

```bash
cd "/home/florian/mes_projets/Defi-IA 2020"
uv venv --python 3.12 .venv
. .venv/bin/activate
uv pip install -r requirements.txt          # cœur CPU (A/B)
# GPU (C/D) : sur desktop 4060 Ti / Kaggle / Colab uniquement
uv pip install -r requirements-gpu.txt
```

### Pipeline (Makefile)
```bash
make eda                 # scan streaming du CSV (sans dépendances)
make data                # raw CSV -> data/interim/{train,test}.parquet
make features            # matrices de features (réseau + texte)
make train-gbm           # LightGBM/CatBoost, objectif MAE
make embeddings          # embeddings de phrase (GPU)
make train-transformer   # fine-tuning encodeur (GPU)
make blend               # blending OOF -> prédiction finale
make submit              # submissions/submission.csv (id,predicted)
```

### Conventions
- Toute logique réutilisable vit dans `src/defia/`, jamais dans les notebooks.
- Config unique : `configs/*.yaml` (jamais de constantes en dur dans le code).
- Validation : **GroupKFold par `link_id`** (thread) partout. Features à fuite (target
  encoding) calculées **dans le fold**.
- MAE toujours rapportée dans l'espace original de la cible.
- Commits fréquents, messages en français, préfixés par le jalon (`A:`, `B:`…).

---

## 2. Décisions structurantes
| Sujet | Décision | Pourquoi |
|---|---|---|
| Métrique de travail | MAE, baseline = médiane (1) | Métrique officielle ; médiane optimale en MAE |
| CV | GroupKFold(`link_id`), 5 folds | Évite la fuite intra-thread |
| Objectif GBM | L1 / Huber / quantile(0.5) | Adapté MAE + queue lourde |
| Route GPU (C/D) | **Kaggle Kernels API** (fallback 4060 Ti) | Seul GPU distant pilotable en autonomie |
| Env Python | `uv` venv, WSL | `python3.12-venv` absent, `uv` auto-suffisant |

---

## 3. Journal de bord

### 2026-07-13 — Setup & EDA
- Décompression `reddit-ut3-ut1.zip` → `data/raw/comments_students.csv` (4 234 970 lignes).
- EDA streaming (`scripts/eda_stream.py`) : 3,22 M train / 1,02 M test ; cible ultra-asymétrique
  (médiane 1, 52 % à 1, min −333, max 6761). Constats complets : `docs/eda_findings.md`.
- Architecture du repo montée (package `src/defia`, docs, configs, Makefile, tests).
- Plan validé avec Florian : jalons A→E en autonomie, route GPU = Kaggle API.
- Environnement `uv` créé (le cœur CPU installé).

### 2026-07-13 — Milestone A : data layer + baselines
- Conversion CSV → parquet (chunké, zstd) : `data/interim/{train,test}.parquet`
  (3 218 512 / 1 016 458 lignes).
- **Constat décisif : split TEMPOREL** (train 1–24 mai, test 25–31 mai, sans recouvrement) →
  validation = holdout temporel 7 jours. `cv.py` mis à jour (`temporal_holdout_indices`,
  `time_series_folds`) ; `configs/default.yaml` scheme=temporal.
- **Baselines (holdout temporel, la barre) :**
  | stratégie | MAE |
  |---|---|
  | prédire 1 (médiane) | **11.907** |
  | médiane/heure | 11.907 |
  | médiane/auteur | 13.253 (pire) |
  | moyenne (12.74) | 20.676 |
- **Enseignement clé** : 52 % des `ups` valent 1 → le cœur ne pèse ~rien en MAE ; les ~11,9 pts
  viennent de la **queue virale**. **Le problème est une régression de queue.** Le feature
  engineering doit cibler « qui devient viral et combien ». Le target encoding naïf par auteur
  dégrade → prévoir lissage bayésien + passé seulement.
- Première soumission (format `id,predicted`, médiane=1) : `submissions/submission_baseline_median.csv`.

### 2026-07-13 (suite) — Milestone B : features réseau/texte + LightGBM
- **Incident infra** : build features initial en un bloc → OOM (WSL ~7,4 Go). Corrigé en deux temps :
  (1) `structural.py` refondu pour factoriser tout de suite les strings en codes entiers
  (`link_code`/`author_code`), matrice purement numérique ; écrit une fois sur disque
  (`_struct.parquet`) et libéré avant la stylométrie (les deux pics mémoire ne se chevauchent
  plus). (2) `features` en streaming chunké (parquet `iter_batches`) avec VADER mis en cache
  module-level. Un bug `click.echo(..., flush=True)` (argument invalide) a aussi fait planter un
  run et a été pris à tort pour un problème mémoire — corrigé.
- **Incident orchestration** : un job d'entraînement enchaîné en arrière-plan est mort
  silencieusement (probable coupure d'interop WSL passagère) sans qu'aucun mécanisme ne le
  détecte pendant ~1h30. Leçon : après tout lancement `run_in_background`, **vérifier activement**
  qu'un vrai PID consomme CPU/RAM (`ps -eo pid,pcpu,pmem,rss,cmd`), ne jamais se fier à un
  `pgrep` qui peut matcher la commande de lancement elle-même.
- **Features construites** : 38 colonnes (18 réseau/structurelles + ~20 stylométrie/sentiment
  VADER), `data/processed/{train,test}_features.parquet`.
- **Premier modèle LightGBM** (objectif `mae`, holdout temporel 7 jours) :
  **MAE = 8.3946** (baseline médiane 11.907, gain **+29.5 %**), best_iter=714.
  → **Dans la zone historique 8–11 dès le premier modèle**, sans encore embeddings/transformer.
- **Importances (top 9/15 = features réseau)** : `thread_size`, `hour`, `pct_in_thread`,
  `n_siblings`, `sibling_rank`, `rank_in_thread`, `age_in_thread`, `thread_n_authors`,
  `n_children` dominent très largement `n_chars`/`n_words` (texte). Confirme l'hypothèse EDA :
  le **network mining porte l'essentiel du signal** à ce stade ; le texte stylométrique est
  secondaire — le gain texte attendu viendra des embeddings/transformer (étapes C/D).
- Soumission : `submissions/submission_gbm_mae.csv`. Rapport : `reports/gbm_gbm_mae.json`.

### 2026-07-13 (suite) — Ablation objectif LightGBM
| Config | MAE holdout | vs baseline (11.907) |
|---|---|---|
| **`mae` (L1), espace direct** | **8.3946** ← champion | +29.5% |
| `huber` | 10.4592 (n'avait pas convergé à 3000 itér.) | +12.2% |
| `mae` + `log1p(target)` | 11.8891 (cassé) | +0.2% |

Le `log1p` non signé cassait sur `ups` négatifs (min -333) → NaN, quasi aucun apprentissage
(best_iter=1). Corrigé en **log signé** (`sign(y)·log1p(|y|)`) dans `gbm.py`, réutilisable plus
tard mais pas prioritaire : **l'objectif `mae` brut domine largement** — la cible en queue lourde
se comporte mieux avec une perte L1 pure qu'après compression log ou Huber (à ce budget
d'itérations). On garde `mae` comme configuration de référence pour la suite.

**Incident process (transparence)** : un job d'entraînement enchaîné a tourné dans le vide
pendant ~1h30 sans qu'aucune vérification ne le détecte (cf. entrée précédente sur l'incident
d'orchestration). Depuis : vérification systématique par `ps -eo pid,pcpu,pmem,rss,cmd` après
tout lancement en arrière-plan, jamais par `pgrep` seul.

### 2026-07-13 (suite) — Target encoding auteur (temporel, sans fuite)
Nouveau module `features/author_target.py` : moyenne lissée (bayésienne, k=20) des `ups`
antérieurs de l'auteur. TRAIN = expanding leave-one-out (strictement antérieur, exclut la ligne
courante) ; TEST = agrégat complet du train par auteur (légitime, tout le train précède tout le
test). Testé sur cas synthétique (assertions exactes) avant le run complet — contrairement à
l'encodage naïf de Milestone A qui dégradait la MAE (13.25), celui-ci est temporellement propre.

| Config | MAE holdout | Δ vs sans author-enc |
|---|---|---|
| GBM `mae` sans author-enc | 8.3946 | — |
| **GBM `mae` + author_hist_mean/count** | **8.3559** | **-0.46%** (léger mais réel) |

`author_hist_mean` (9e) et `author_hist_count_log` (12e) entrent dans le top 15 des importances,
sans dominer — le signal de timing/thread reste prépondérant. Nouveau champion : **MAE 8.3559**.
Commande : `make author-encoding` puis `make train-gbm` (merge automatique si les fichiers
`*_author_enc.parquet` existent).

### 2026-07-13 (suite) — TF-IDF abandonné pour ce jalon ; clôture Milestone B
Implémenté `features/text.py` (`HashingVectorizer` + `TruncatedSVD`, stateless/mémoire-léger
par design, testé par smoke test unitaire avant le run complet — le code est correct) et la
commande `tfidf-features`. Deux tentatives :

- **128 composantes SVD** : build features OK, mais le fichier dense obtenu (2,2 Go train) a fait
  monter LightGBM à 6,3 Go RSS pendant la construction du dataset (WSL plafonné à 7,4 Go) →
  process tué (exit 15).
- **48 composantes SVD** (réduction pour rentrer dans le budget mémoire) : **3 tentatives
  consécutives** interrompues par une **instabilité de l'hôte Windows lui-même** (WSL puis
  parfois PowerShell entier injoignables pendant plusieurs dizaines de secondes, y compris hors
  de tout pic mémoire mesuré côté WSL) — pas un bug du code, confirmé par : (a) smoke test
  unitaire passant, (b) les runs interrompus avaient progressé normalement sans erreur Python
  ni trace OOM avant la coupure, (c) le dernier fichier partiel (450k/3,2M lignes) ne montrait
  aucune anomalie.

**Décision : abandon du TF-IDF pour ce jalon**, conformément au garde-fou fixé (ne pas boucler
indéfiniment sur un échec d'infrastructure hors de notre contrôle). Le code reste dans le repo,
inchangé et réutilisable tel quel sur une machine plus stable (desktop 4060 Ti, Kaggle) —
`make tfidf-features` puis `make train-gbm` avec `data/processed/*_tfidf.parquet` présents.

**Milestone B clos sur le champion : MAE = 8.3559** (holdout temporel), soit **-29.8% vs la
baseline** (11.907) — features réseau/structurelles + stylométrie/sentiment + target encoding
auteur temporellement propre. Solidement dans la zone historique 8-11 communiquée par Florian.

**Bilan des essais du jalon :**
| Config | MAE | Retenu |
|---|---|---|
| Baseline (médiane) | 11.907 | — |
| GBM `mae`, réseau+texte | 8.3946 | base |
| GBM `mae`, + target encoding auteur | **8.3559** | ✅ **champion** |
| GBM `huber` | 10.4592 | non (pire) |
| GBM `mae` + log1p signé | (non re-testé après fix) | — |
| GBM `mae` + TF-IDF 48-dim | — | abandonné (infra) |

### 2026-07-13/14 (nuit) — Milestone C/D sur Kaggle : impasses, décision, finalisation E
**Kaggle utilisé** (compte flosal) : dataset `flosal/defia-reddit-text` uploadé, kernels
`flosal/defia-embeddings` et `flosal/defia-transformer` poussés et exécutés. Bilan honnête des
impasses rencontrées :

- **Embeddings (C) en CPU** : e5-small sur 4,2 M lignes > 7 h sans finir → impraticable sur le
  CPU partagé Kaggle. Abandonné pour ce soir.
- **Transformer (D) — 4 échecs successifs**, chacun diagnostiqué :
  1. GPU **P100 (sm_60)** incompatible avec le torch pré-installé Kaggle (supporte sm_70+).
  2. Fix tenté : install torch PyPI (compatible Pascal) + ré-exécution → passe le GPU, mais…
  3. **ModernBERT** absent de la version `transformers` de Kaggle (exige ≥4.48).
  4. Bascule DistilBERT → l'install forcé de torch 2.4.1 **casse `transformers`**
     (`Could not import module 'DistilBertModel'`) : réinstaller torch désaligne tout l'env.

**Décision** : arrêter d'itérer sur le GPU Kaggle (puits sans fond ce soir). Le transformer +
les embeddings tourneront sur la **4060 Ti locale de Florian** (sm_89, torch/transformers natifs,
minutes, zéro friction) — les kernels/modules sont prêts et réutilisables. Milestone E finalisé
cette nuit **avec les GBM seuls** (blend de variantes) pour livrer un pipeline complet de bout en
bout ; le transformer sera intégré au blend demain.

### 2026-07-14 — Milestone E : TF-IDF (gain), blend final, clôture
- **TF-IDF fusionné** (HashingVectorizer word 1-2 + SVD 32d, sur CPU) → GBM `mae` à 72 features :
  **MAE holdout = 8.2841** vs 8.3559 sans → **le signal texte lexical apporte +0.07 de MAE**
  (nouveau meilleur modèle). Confirme que la stylométrie seule sous-exploitait le texte.
- **Blend final** (poids optimisant la MAE) : gbm_tfidf 0.878 + gbm_champion 0.122 →
  **MAE = 8.2826** (gain marginal, les 2 GBM sont corrélés). Soumission :
  `submissions/submission_final.csv` (1 016 458 lignes, format OK, médiane 1.08, queue jusqu'à 4621).
- Variante `quantile` abandonnée (MAE 8.356 ≈ champion, aucune diversité).
- **GPU distant Kaggle : mur** ce soir — P100 sm_60 incompatible + réinstall torch casse
  `transformers` + quota GPU (2 sessions max) saturé. Le transformer/embeddings tourneront sur
  GPU récent quand dispo (code prêt). Rapport complet : `reports/rapport.md`.
- **Bilan A→E : baseline 11.907 → 8.283 (−30.4 %)**, dans la zone historique 8–11.

### 2026-07-14 — Vitesse supérieure : feature engineering avancé (network mining sérieux)
Recadrage Florian : les techniques étaient trop basiques. On empile des features avancées et on
mesure chaque gain (holdout temporel, mode `--eval-only` pour itérer vite).
- **Features de contexte** (`features/context.py`, 9 feats) : écart temporel à la réponse
  (`time_gap_to_parent`), parent-carrefour (`parent_n_children`, `parent_rank`, `parent_depth`),
  réponse à soi-même, **vélocité du thread**, position dans le cycle de vie du fil
  (`arrival_frac`), **nb de commentaires déjà postés par l'auteur dans ce fil**.
- **Dynamique d'auteur enrichie** (`features/author_target.py::build_author_dynamics`, 6 feats,
  temporellement propre) : moyenne/écart-type/max/fraction virale/fraction downvote historiques.
- **Résultat GBM 87 features : MAE = 8.2667** vs 8.284 (TF-IDF seul) → **+0.017**, +30.6 % vs
  baseline. `author_prior_in_thread` et `n_children` (carrefour) rankent haut.
- En cours : **node2vec** (embeddings du graphe auteur→auteur, `features/graph.py`) + **CatBoost**
  (haute cardinalité native). Objectif : passer nettement sous 8.2.

<!-- Prochaines entrées ajoutées à chaque jalon : résultats MAE, choix, ablations. -->
