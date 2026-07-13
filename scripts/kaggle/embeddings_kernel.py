"""Kernel Kaggle (GPU) — extraction d'embeddings de phrase pour les commentaires Reddit.

Autonome (pas de dépendance à `defia` : les kernels Kaggle n'ont pas notre package installé).
Entrée : dataset Kaggle attaché contenant `train.parquet` et `test.parquet` (colonnes id, body).
Sortie (écrite dans /kaggle/working/, récupérée via `kaggle kernels output`) :
    train_emb.parquet, test_emb.parquet — colonnes id, emb_0..emb_{n_components-1} (float32).

Stratégie mémoire (le kernel a plus de RAM que notre laptop mais on reste prudent, 4,2M lignes) :
    1. Encoder un échantillon (SAMPLE_SIZE) pour ajuster une TruncatedSVD (384 -> N_COMPONENTS).
    2. Encoder le reste par batches, réduire immédiatement via la SVD, écrire en streaming —
       on ne garde jamais l'ensemble des embeddings 384-dim en mémoire.
"""
import glob
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD

MODEL_NAME = "intfloat/e5-small-v2"   # petit, rapide, bon compromis MTEB (2024)
N_COMPONENTS = 64
SAMPLE_SIZE = 200_000
BATCH_SIZE = 512
SEED = 42

INPUT_DIR = next(
    (d for d in glob.glob("/kaggle/input/*") if os.path.exists(os.path.join(d, "train.parquet"))),
    "/kaggle/input/defia-reddit-text",
)
OUT_DIR = "/kaggle/working"


def load_texts(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["id", "body"])
    df["body"] = df["body"].fillna("")
    return df


def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    # e5 attend un préfixe "passage: " pour les textes à indexer (asymétrique query/passage).
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(
        prefixed, batch_size=BATCH_SIZE, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)


def main() -> None:
    print(f"[kernel] modèle={MODEL_NAME} device=cuda si dispo")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    train_path = os.path.join(INPUT_DIR, "train.parquet")
    test_path = os.path.join(INPUT_DIR, "test.parquet")

    print("[kernel] échantillon pour ajuster la SVD...")
    tr_sample = load_texts(train_path).sample(n=SAMPLE_SIZE, random_state=SEED)
    sample_emb = encode(model, tr_sample["body"].tolist())
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED)
    svd.fit(sample_emb)
    print(f"[kernel] SVD ajustée, variance expliquée cumulée="
          f"{svd.explained_variance_ratio_.sum():.3f}")
    del tr_sample, sample_emb

    cols = ["id"] + [f"emb_{i}" for i in range(N_COMPONENTS)]
    for split, path in (("train", train_path), ("test", test_path)):
        df = load_texts(path)
        out_path = os.path.join(OUT_DIR, f"{split}_emb.parquet")
        writer = None
        n = len(df)
        print(f"[kernel] encodage {split} ({n:,} lignes)...")
        for start in range(0, n, BATCH_SIZE * 20):
            chunk = df.iloc[start:start + BATCH_SIZE * 20]
            emb = encode(model, chunk["body"].tolist())
            reduced = svd.transform(emb).astype(np.float32)
            feat = pd.DataFrame(reduced, columns=cols[1:])
            feat.insert(0, "id", chunk["id"].to_numpy())
            tbl = pa.Table.from_pandas(feat, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            print(f"    {split}: {min(start + len(chunk), n):,}/{n:,}")
        writer.close()
        print(f"[kernel] écrit {out_path}")

    print("[kernel] terminé.")


if __name__ == "__main__":
    main()
