#!/usr/bin/env bash
# Poll un kernel Kaggle déjà lancé jusqu'à complétion, puis rapatrie ses sorties.
# Usage : bash scripts/kaggle/poll_pull.sh <kernel_id> <output_dir>
# Ex.   : bash scripts/kaggle/poll_pull.sh flosal/defia-embeddings data/processed
set +u
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null

KID="${1:?usage: poll_pull.sh <kernel_id> <out_dir>}"
OUT="${2:-data/processed}"

for i in $(seq 1 240); do   # ~2h max
    RAW="$(kaggle kernels status "$KID" 2>/dev/null || true)"
    ST="$(echo "$RAW" | grep -oiE '(complete|error|running|queued|cancel)' | head -1 | tr '[:upper:]' '[:lower:]')"
    ST="${ST:-unknown}"
    echo "[$i] $(date +%H:%M:%S) status=$ST"
    if [ "$ST" = "complete" ] || [ "$ST" = "error" ] || [ "$ST" = "cancel" ]; then
        break
    fi
    sleep 30
done

echo "FINAL_STATUS=$ST"
if [ "$ST" = "complete" ]; then
    echo "[poll] rapatriement des sorties -> $OUT"
    kaggle kernels output "$KID" -p "$OUT" 2>&1
    ls -la "$OUT"/*.parquet 2>/dev/null || true
fi
echo "POLL_DONE"
