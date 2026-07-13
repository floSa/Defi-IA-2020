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

<!-- Prochaines entrées ajoutées à chaque jalon : résultats MAE, choix, ablations. -->
