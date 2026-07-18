"""Extraction d'embeddings de phrase en local sur GPU (RTX 4060 Ti).

Version locale du kernel Kaggle `scripts/kaggle/embeddings_kernel.py`, qui n'a jamais pu
tourner faute de GPU compatible (P100/sm_60). Écrit data/processed/{split}_emb.parquet
(colonnes id, emb_0..emb_{N_COMPONENTS-1}), format déjà attendu par `defia train-gbm`.

Optimisations vs le kernel : fp16, tri par longueur de texte (réduit le padding), et
streaming SVD pour ne jamais tenir les 4,2M x 384 floats en mémoire.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD

MODEL_NAME = os.environ.get("EMB_MODEL", "intfloat/e5-small-v2")
PREFIX = os.environ.get("EMB_PREFIX", "passage: ")
N_COMPONENTS = 64
SAMPLE_SIZE = 200_000
BATCH_SIZE = 512
CHUNK = 400_000
SEED = 42
SUFFIX = os.environ.get("EMB_SUFFIX", "emb")


def load_model() -> SentenceTransformer:
    m = SentenceTransformer(MODEL_NAME, device="cuda", trust_remote_code=True)
    m.max_seq_length = 128
    return m.half()


def encode(model, texts: list[str]) -> np.ndarray:
    return model.encode([PREFIX + t for t in texts], batch_size=BATCH_SIZE,
                        show_progress_bar=False, convert_to_numpy=True,
                        normalize_embeddings=True).astype(np.float32)


def encode_sorted(model, texts: list[str]) -> np.ndarray:
    """Encode en triant par longueur (moins de padding) puis restaure l'ordre initial."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = encode(model, [texts[i] for i in order])
    restored = np.empty_like(out)
    restored[order] = out
    return restored


def main() -> None:
    t_start = time.time()
    print(f"[emb] modèle={MODEL_NAME} gpu={torch.cuda.get_device_name(0)}", flush=True)
    model = load_model()

    print(f"[emb] ajustement SVD sur échantillon (n={SAMPLE_SIZE:,})...", flush=True)
    sample = pd.read_parquet("data/interim/train.parquet", columns=["body"]).sample(
        n=SAMPLE_SIZE, random_state=SEED)
    emb = encode_sorted(model, sample["body"].fillna("").astype(str).tolist())
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED).fit(emb)
    print(f"[emb] SVD ok, variance expliquée={svd.explained_variance_ratio_.sum():.3f} "
          f"({time.time()-t_start:.0f}s)", flush=True)
    del sample, emb

    cols = [f"emb_{i}" for i in range(N_COMPONENTS)]
    for split in ("train", "test"):
        src = f"data/interim/{split}.parquet"
        out = f"data/processed/{split}_{SUFFIX}.parquet"
        pf = pq.ParquetFile(src)
        writer, done = None, 0
        for batch in pf.iter_batches(batch_size=CHUNK, columns=["id", "body"]):
            b = batch.to_pandas()
            vecs = svd.transform(encode_sorted(model, b["body"].fillna("").astype(str).tolist()))
            feat = pd.DataFrame(vecs.astype(np.float32), columns=cols)
            feat.insert(0, "id", b["id"].to_numpy())
            tbl = pa.Table.from_pandas(feat, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            done += len(b)
            print(f"    {split}: {done:,} ({time.time()-t_start:.0f}s)", flush=True)
        writer.close()
        print(f"[emb] écrit {out} ({done:,} lignes)", flush=True)
    print(f"[emb] terminé en {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
