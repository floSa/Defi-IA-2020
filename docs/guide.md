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

<!-- Prochaines entrées ajoutées à chaque jalon : résultats MAE, choix, ablations. -->
