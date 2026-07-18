"""Benchmark d'encodage GPU : mesure le débit réel avant de lancer sur 4,2M lignes.

Compare e5-small-v2 (2023, 33M params, 384d) et gte-modernbert-base (2025, 149M, 768d)
sur un échantillon, pour choisir le modèle tenable dans le budget temps.
"""
import sys
import time

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
MODELS = [("intfloat/e5-small-v2", "passage: "), ("Alibaba-NLP/gte-modernbert-base", "")]

df = pd.read_parquet("data/interim/train.parquet", columns=["body"]).head(N)
texts = df["body"].fillna("").astype(str).tolist()
# Tri par longueur : réduit fortement le padding dans les batches (gain réel 1.5-2x).
texts = sorted(texts, key=len)
print(f"n={len(texts):,} | longueur médiane={len(texts[len(texts)//2])} car.")

for name, prefix in MODELS:
    try:
        m = SentenceTransformer(name, device="cuda", trust_remote_code=True)
        m.max_seq_length = 128  # commentaires Reddit : courts
        m = m.half()
        batch = [prefix + t for t in texts]
        m.encode(batch[:512], batch_size=256, show_progress_bar=False)  # warmup
        torch.cuda.synchronize()
        t0 = time.time()
        m.encode(batch, batch_size=256, show_progress_bar=False,
                 convert_to_numpy=True, normalize_embeddings=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        rate = len(texts) / dt
        print(f"{name}: {rate:,.0f} textes/s -> 4.23M en {4_235_000/rate/60:.1f} min "
              f"| dim={m.get_sentence_embedding_dimension()}")
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{name}: ECHEC -> {type(e).__name__}: {e}")
