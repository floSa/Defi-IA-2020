# Orchestration Défi IA 2020. Cibles = étapes du pipeline (cf. docs/plan.md).
# Chaque cible appelle la CLI du package (src/defia/cli.py), paramétrée par configs/*.yaml.

PY ?= python
CONFIG ?= configs/default.yaml

.PHONY: help setup eda data baseline features train-gbm train-transformer embeddings blend submit test clean

help:
	@echo "Cibles :"
	@echo "  setup             installe le package en editable (CPU)"
	@echo "  eda               scan streaming du CSV (sans dépendances)"
	@echo "  data              raw CSV -> data/interim/{train,test}.parquet"
	@echo "  features          construit les matrices de features (réseau + texte)"
	@echo "  train-gbm         entraîne le LightGBM/CatBoost (objectif MAE)"
	@echo "  embeddings        extrait les embeddings de phrase (GPU)"
	@echo "  train-transformer fine-tune l'encodeur body->ups (GPU)"
	@echo "  blend             blending OOF des modèles -> prédiction finale"
	@echo "  submit            écrit submissions/submission.csv (id,predicted)"
	@echo "  test              pytest"

setup:
	$(PY) -m pip install -e ".[dev]"

eda:
	$(PY) scripts/eda_stream.py data/raw/comments_students.csv

data:
	$(PY) -m defia.cli data --config $(CONFIG)

baseline:
	$(PY) -m defia.cli baseline --config $(CONFIG)

features:
	$(PY) -m defia.cli features --config $(CONFIG)

author-encoding:
	$(PY) -m defia.cli author-encoding --config $(CONFIG)

tfidf-features:
	$(PY) -m defia.cli tfidf-features --config $(CONFIG)

train-gbm:
	$(PY) -m defia.cli train-gbm --config $(CONFIG)

embeddings:
	$(PY) -m defia.cli embeddings --config $(CONFIG)

train-transformer:
	$(PY) -m defia.cli train-transformer --config $(CONFIG)

blend:
	$(PY) -m defia.cli blend --config $(CONFIG)

submit:
	$(PY) -m defia.cli submit --config $(CONFIG)

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/interim/* data/processed/* models/* reports/figures/*
