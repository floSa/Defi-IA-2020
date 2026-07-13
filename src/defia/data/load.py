"""Chargement du CSV brut et conversion train/test en parquet.

Points d'attention (issus de l'EDA, cf. docs/eda_findings.md) :
  * ``ups`` NaN  ->  ligne de test ; sinon train.
  * Le ``body`` contient du markdown brut et des entités HTML échappées (``&lt; &gt; &amp;``).
    On conserve le texte brut (le markdown est un signal) ; l'unescape se fait au feature-time.
  * Conversion **chunkée** CSV -> parquet pour rester léger en RAM (4,2 M lignes).

Sortie : ``data/interim/train.parquet`` et ``data/interim/test.parquet``.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Colonnes du CSV et dtypes de lecture (pandas). ups en float (NaN = test).
_STR_COLS = ["subreddit_id", "link_id", "name", "subreddit", "id", "author", "body", "parent_id"]
_READ_DTYPES = {c: "string" for c in _STR_COLS}
_READ_DTYPES["created_utc"] = "int64"
_READ_DTYPES["ups"] = "float64"

# Schéma parquet cible (compact). ups en float32 (NaN sur test), created_utc int64.
_SCHEMA = pa.schema(
    [("created_utc", pa.int64()), ("ups", pa.float32())]
    + [(c, pa.string()) for c in _STR_COLS]
)
_ORDER = [f.name for f in _SCHEMA]


def unescape_body(s: pd.Series) -> pd.Series:
    """Décode les entités HTML (&lt; &gt; &amp; ...). Utilisé au feature-time, pas au stockage."""
    return s.fillna("").map(html.unescape)


def build_train_test(
    csv_path: str | Path,
    out_dir: str | Path,
    chunksize: int = 500_000,
) -> dict[str, int]:
    """Convertit le CSV brut en deux parquet (train/test) en flux.

    train = lignes où ``ups`` est renseigné ; test = lignes où ``ups`` est NaN.
    Retourne le décompte des lignes écrites.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"

    train_writer = pq.ParquetWriter(train_path, _SCHEMA, compression="zstd")
    test_writer = pq.ParquetWriter(test_path, _SCHEMA, compression="zstd")
    n_train = n_test = 0
    try:
        reader = pd.read_csv(
            csv_path,
            dtype=_READ_DTYPES,
            chunksize=chunksize,
            na_values=[],  # ne pas transformer les chaînes vides en NaN (sauf ups, géré à part)
            keep_default_na=True,
        )
        for i, chunk in enumerate(reader):
            chunk = chunk[_ORDER]
            is_test = chunk["ups"].isna()
            train_df = chunk[~is_test]
            test_df = chunk[is_test]
            if len(train_df):
                train_writer.write_table(pa.Table.from_pandas(train_df, schema=_SCHEMA, preserve_index=False))
                n_train += len(train_df)
            if len(test_df):
                test_writer.write_table(pa.Table.from_pandas(test_df, schema=_SCHEMA, preserve_index=False))
                n_test += len(test_df)
            print(f"  chunk {i}: +{len(train_df)} train, +{len(test_df)} test "
                  f"(cumul {n_train}/{n_test})", flush=True)
    finally:
        train_writer.close()
        test_writer.close()

    return {"train": n_train, "test": n_test}


def load_split(interim_dir: str | Path, split: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Relit un split parquet (``train`` ou ``test``), éventuellement un sous-ensemble de colonnes."""
    path = Path(interim_dir) / f"{split}.parquet"
    return pd.read_parquet(path, columns=columns)
