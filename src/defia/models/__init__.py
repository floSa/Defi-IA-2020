"""Modèles : baselines, GBM, transformer, blending.

    baseline.py     constante médiane + médianes par groupe (plancher MAE)
    gbm.py          LightGBM / CatBoost, objectif L1/Huber/quantile
    transformer.py  fine-tuning encodeur body->ups (GPU) — étape D
    blend.py        blending des prédictions OOF (ridge / lgbm / nnls)

Voir docs/plan.md §3 pour le tableau des modèles et le traitement de l'asymétrie (MAE).
"""
