#!/usr/bin/env bash
# Rassemble toutes les matrices de features dans un dossier pour upload en dataset Kaggle.
set -e
cd "$(dirname "$0")/../.."
DEST="scripts/kaggle/features_dataset"
rm -rf "$DEST"
mkdir -p "$DEST"
for f in features author_enc context author_dyn parentenc interactions; do
    cp "data/processed/train_$f.parquet" "$DEST/" 2>/dev/null || true
    cp "data/processed/test_$f.parquet" "$DEST/" 2>/dev/null || true
done
# tfidf peut être dans data/processed ou mis de côté dans /tmp/hold_tfidf
cp data/processed/train_tfidf.parquet data/processed/test_tfidf.parquet "$DEST/" 2>/dev/null || \
    cp /tmp/hold_tfidf/train_tfidf.parquet /tmp/hold_tfidf/test_tfidf.parquet "$DEST/" 2>/dev/null || true
echo "Contenu de $DEST :"
ls -1 "$DEST"
echo "Taille : $(du -sh "$DEST" | cut -f1)"
