#!/usr/bin/env bash
# Orchestration Kaggle : export data -> push dataset -> push kernel(s) -> poll -> pull résultats.
#
# Prérequis : ~/.kaggle/kaggle.json présent (chmod 600) et `kaggle` installé dans le venv.
# Usage : bash scripts/kaggle/prepare_and_push.sh embeddings|transformer
set -euo pipefail
cd "$(dirname "$0")/../.."
. .venv/bin/activate

KIND="${1:?usage: prepare_and_push.sh embeddings|transformer}"
KDIR="scripts/kaggle"
DATASET_DIR="$KDIR/dataset"
DATASET_SLUG="defia-reddit-text"

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERREUR: ~/.kaggle/kaggle.json absent. Place ton token Kaggle d'abord." >&2
    exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"
USERNAME="$(python -c 'import json; print(json.load(open("'"$HOME"'/.kaggle/kaggle.json"))["username"])')"
echo "[kaggle] username=$USERNAME"

# 1) Export léger (id, body, created_utc[, ups]) — pas le CSV brut complet.
if [ ! -f "$DATASET_DIR/train.parquet" ]; then
    echo "[kaggle] export des données..."
    PYTHONPATH=src python -m defia.cli kaggle-export --config configs/default.yaml --out "$DATASET_DIR"
fi

# 2) Dataset Kaggle (create si absent, version sinon)
cat > "$DATASET_DIR/dataset-metadata.json" <<EOF
{"title": "$DATASET_SLUG", "id": "$USERNAME/$DATASET_SLUG", "licenses": [{"name": "CC0-1.0"}]}
EOF
if kaggle datasets list -m --user "$USERNAME" 2>/dev/null | grep -q "$DATASET_SLUG"; then
    echo "[kaggle] dataset existe -> version"
    kaggle datasets version -p "$DATASET_DIR" -m "update $(date -u +%Y-%m-%dT%H:%M:%SZ)" -d
else
    echo "[kaggle] dataset absent -> create"
    kaggle datasets create -p "$DATASET_DIR" -d
fi

# 3) Kernel push (substitue USERNAME dans le kernel-metadata correspondant)
META_SRC="$KDIR/kernel-metadata-$KIND.json"
META_DST="$KDIR/kernel-metadata.json"
sed "s/USERNAME/$USERNAME/g" "$META_SRC" > "$META_DST"
echo "[kaggle] push kernel $KIND..."
kaggle kernels push -p "$KDIR"

# 4) Poll jusqu'à complete/error (kernel id = username/defia-<kind>)
KERNEL_ID="$USERNAME/defia-$KIND"
echo "[kaggle] attente de la fin d'exécution ($KERNEL_ID)..."
for i in $(seq 1 180); do   # jusqu'à ~90 min (30s * 180)
    STATUS="$(kaggle kernels status "$KERNEL_ID" 2>/dev/null | grep -oE '"(complete|error|running|queued)"' | tr -d '"' || echo "unknown")"
    echo "  [$i] status=$STATUS"
    if [ "$STATUS" = "complete" ] || [ "$STATUS" = "error" ]; then
        break
    fi
    sleep 30
done

if [ "$STATUS" != "complete" ]; then
    echo "ERREUR: kernel terminé avec status=$STATUS (voir logs Kaggle)." >&2
    exit 1
fi

# 5) Rapatrie les sorties
OUT_DIR="data/processed"
mkdir -p "$OUT_DIR"
echo "[kaggle] rapatriement des sorties -> $OUT_DIR"
kaggle kernels output "$KERNEL_ID" -p "$OUT_DIR"
echo "[kaggle] terminé. Fichiers dans $OUT_DIR"
