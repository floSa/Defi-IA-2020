"""Défi IA 2020 — prédiction des upvotes Reddit (MAE), text + network mining.

Package structuré en couches :
    data/        chargement, split train/test, nettoyage
    features/    structural (réseau), stylometric, text, temporal
    models/      baseline, gbm, transformer, blend
    evaluation/  métriques (MAE) et schéma de validation (GroupKFold par thread)

Le pipeline est orchestré via ``defia.cli`` (voir le Makefile) et paramétré par
``configs/*.yaml`` (voir ``defia.config``).
"""

__version__ = "0.1.0"
