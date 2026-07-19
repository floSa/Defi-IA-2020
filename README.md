# Défi IA 2020 — Prédiction des upvotes Reddit (revisité 2026)

Reconstruction propre et modernisée d'un challenge Kaggle (TSE / Université Paul Sabatier,
2020) : **prédire le score `ups` d'un commentaire AskReddit (mai 2015)**, métrique **MAE**,
en combinant **text mining** et **network mining**.

> État du projet : **squelette + plan**. Le code de modélisation démarre après validation du
> [plan de travail](docs/plan.md).

## Démarrage rapide
```bash
# 1. Environnement (CPU)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                              # ou: pip install -r requirements.txt

# 2. Données : placer comments_students.csv dans data/raw/ (déjà fait si extrait du zip)
make data        # -> data/interim/{train,test}.parquet

# 3. Pipeline (à venir, cf. plan)
make features
make train-gbm
make blend
make submit
```

## Documentation
| Fichier | Contenu |
|---|---|
| [docs/challenge_brief.md](docs/challenge_brief.md) | Définition officielle de la tâche, métrique, format |
| [docs/eda_findings.md](docs/eda_findings.md) | Constats EDA sur le corpus complet (chiffres) |
| [docs/plan.md](docs/plan.md) | **Plan de modélisation à valider** |
| [docs/compute_strategy.md](docs/compute_strategy.md) | Local vs GPU distant (desktop / Kaggle / Colab) |
| [docs/preconisations.md](docs/preconisations.md) | **⚠️ À lire en premier** — l'écart validation / Kaggle et ce qu'il implique |
| [reports/rapport.md](reports/rapport.md) | Résultats détaillés, ablations, pièges de mesure |

## État au 19 juillet 2026
| | MAE |
|---|---|
| Notre holdout temporel | 7,8695 |
| **Score privé Kaggle** | **8,0878** |
| 1ᵉʳ du classement 2020 | 7,8271 |

**Notre validation surestime de 0,22** — plus que tous les gains cumulés de la dernière session.
La priorité n'est donc plus d'optimiser le modèle mais de **réparer l'instrument de mesure**
(validation multi-fenêtres). Détails et plan d'action : [docs/preconisations.md](docs/preconisations.md).

## Architecture
```
Defi-IA 2020/
├── data/            raw · interim · processed · external   (gitignored)
├── docs/            brief, EDA, plan, stratégie de calcul
├── src/defia/       package : data · features · models · evaluation
│   ├── config.py    chemins & hyperparamètres (piloté par configs/*.yaml)
│   ├── data/        chargement, split, nettoyage
│   ├── features/    structural (réseau) · stylometric · text · temporal
│   ├── models/      baseline · gbm · transformer · blend
│   └── evaluation/  métriques MAE · schéma de CV (GroupKFold thread)
├── configs/         hyperparamètres par expérience (yaml)
├── scripts/         entrées CLI / jobs (local, Kaggle, Colab)
├── notebooks/       exploration
├── models/          artefacts entraînés                    (gitignored)
├── submissions/     fichiers id,predicted
├── reports/         figures, ablations, rapport
└── Makefile         orchestration de bout en bout
```

## Idée directrice (issue de l'EDA)
La MAE est minimisée par la **médiane** ; ici 52 % des commentaires valent exactement 1.
« Prédire 1 » est donc un baseline redoutable — le gain vient de savoir prédire la **queue
virale** et les **downvotes**. Toute l'ingénierie de features et le choix des objectifs (L1 /
Huber / quantile) découlent de ce constat. Détails : [docs/eda_findings.md](docs/eda_findings.md).

## Reproductibilité
Environnement d'exécution **agnostique** (local CPU / desktop GPU / Kaggle / Colab), piloté par
`configs/*.yaml` et le `Makefile`. Artefacts en parquet/json/csv, rapatriables par git, MCP
Google Drive ou Kaggle output. Voir [docs/compute_strategy.md](docs/compute_strategy.md).
