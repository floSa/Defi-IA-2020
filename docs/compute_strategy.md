# Stratégie de calcul — local vs GPU distant

## Le matériel disponible
| Machine | CPU | RAM | GPU | Rôle |
|---|---|---|---|---|
| Laptop (session actuelle) | — | ? | **aucun** | Dev, EDA, features, GBM légers |
| Desktop perso | Ryzen 9600X | 64 Go | **RTX 4060 Ti (8 Go VRAM)** | Features lourdes, GBM, fine-tuning transformer *small* |
| GPU distant | — | — | T4/L4/A100 selon offre | Transformers plus lourds, gros embeddings |

## Ce qui tourne où
- **Étapes A/B (nettoyage, features structurelles + stylométriques, TF-IDF, LightGBM/CatBoost)**
  → **100 % en local (CPU)**. 3,2 M lignes × ~200 features tiennent en RAM ; LightGBM en
  histogram mode est rapide sur CPU. Pas besoin de GPU.
- **Étapes C/D (embeddings de phrase, fine-tuning d'un encodeur type BERT)** → **GPU requis**.
  8 Go de VRAM suffisent pour DistilBERT / ModernBERT-base / DeBERTa-v3-small en fp16 +
  gradient accumulation, et pour l'extraction d'embeddings par batch. Un encodeur *large*
  ou un fine-tuning rapide sur 3,2 M lignes gagnent à passer sur T4/L4/A100 distant.

## « Piloter Colab à distance via un MCP » — ce qui est faisable ou non
**Honnêtement :** il n'existe pas de MCP officiel/robuste qui exécute du code sur un runtime
Colab de bout en bout sans intervention humaine. Colab n'expose pas d'API publique stable
d'exécution. Voici les routes réellement exploitables, de la plus automatisable à la moins :

### Route 1 — Desktop 4060 Ti (recommandée par défaut pour C/D)
- **Pour** : gratuit, privé, VRAM dédiée, aucune limite de quota, données déjà locales.
- **Contre** : je ne peux pas la déclencher depuis cette session (autre machine). Il faut soit
  y lancer Claude Code, soit exécuter `make train-transformer` à la main puis me rapatrier les
  artefacts.
- **Boucle** : repo synchronisé (git) → tu lances la cible make → artefacts (OOF preds, métriques,
  modèle) écrits dans `models/` et `reports/`.

### Route 2 — Kaggle Kernels API (la plus automatisable à distance)
- Kaggle offre un **GPU headless** (T4×2 / P100, ~30 h/semaine gratuites) piloté par CLI :
  `kaggle kernels push` (envoie un script/notebook), `kernels status` (poll), `kernels output`
  (rapatrie les sorties). **C'est le seul GPU distant que je peux piloter en semi-autonomie**
  (pousser, sonder l'état, récupérer les résultats) sans clic humain à chaque run.
- Données : on publie `comments_students.csv` (ou une version pré-features) comme *Dataset Kaggle
  privé* qu'on attache au kernel. C'est cohérent : c'est une compétition Kaggle à l'origine.
- **Pour** : automatisable, reproductible, gratuit. **Contre** : quota hebdo, setup token API,
  runs limités à ~9 h/12 h.

### Route 3 — Colab + Google Drive (via le MCP Drive déjà connecté)
- Un connecteur **Google Drive** est disponible dans cette session (lecture/écriture de fichiers).
  Il permet de **pousser du code/données vers Drive et de lire les artefacts en retour** — mais
  **il n'exécute rien**. L'exécution reste un clic humain dans Colab.
- **Boucle** : notebook Colab (ouvert depuis GitHub) monte Drive → tu exécutes → il écrit
  `oof_transformer.parquet`, `metrics.json`, `submission.csv` dans Drive → je les lis via le MCP
  Drive et j'enchaîne le blending en local.
- **Pour** : A100/L4 dispo sur Colab Pro, interactif. **Contre** : pas d'exécution automatique de
  ma part ; sessions éphémères ; quotas.

### Ce qui n'est PAS faisable
- Que je **démarre et exécute seul un runtime GPU Colab** de bout en bout. Il y a toujours un
  humain dans la boucle pour l'exécution Colab (Route 3) ou un déclenchement manuel sur le
  desktop (Route 1). Seule la Route 2 (Kaggle) approche l'exécution distante automatisée.

## Principe d'architecture qui découle de tout ça
On rend le code **agnostique à l'environnement d'exécution** : mêmes modules `src/defia`,
paramétrés par `configs/*.yaml` et une variable d'env `DEFIA_DEVICE`/`DEFIA_ENV`. Un même
script tourne en local CPU, sur le desktop GPU, dans un kernel Kaggle ou dans Colab. Les
**artefacts transitent par des formats simples** (`parquet`/`json`/`csv`) rapatriables via
git, Drive (MCP) ou Kaggle output. Le **blending final et la soumission se font toujours en
local**, à partir des prédictions OOF/test produites par chaque brique où qu'elle ait tourné.
