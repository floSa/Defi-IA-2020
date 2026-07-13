"""Fine-tuning d'un encodeur ``body -> ups`` — étape D (GPU).

Encodeur moderne (ModernBERT-base / DeBERTa-v3-small / DistilBERT) avec tête de régression et
loss L1/Huber. Tient sur 8 Go de VRAM (fp16 + gradient accumulation) ; gagne à passer sur
GPU distant (Kaggle / Colab) pour la vitesse. Produit des prédictions OOF pour le blending.

Ce module n'importe torch/transformers que si appelé (dépendances ``.[gpu]``), afin que le
reste du pipeline reste installable et exécutable sur le laptop CPU.
Voir docs/compute_strategy.md pour le choix de la machine d'exécution.
TODO (post-validation) : implémenter le fine-tuning et l'inférence.
"""
from __future__ import annotations


def train_transformer_oof(texts, y, groups, cfg):
    raise NotImplementedError("À implémenter après validation du plan (étape D, GPU).")
