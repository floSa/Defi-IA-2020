#!/usr/bin/env bash
# Upload le dataset de features + push le kernel GBM CPU (30 Go RAM) qui entraîne le LightGBM
# complet sans contrainte mémoire, puis poll + rapatrie les sorties.
set -eo pipefail
cd "$(dirname "$0")/../.."
set +u; . .venv/bin/activate; set -u

USER=$(python -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])')
DSDIR="scripts/kaggle/features_dataset"
KDIR="scripts/kaggle"

# 1) Dataset des features (create ou version, tolère l'erreur JSON de finalisation)
cat > "$DSDIR/dataset-metadata.json" <<EOF
{"title": "defia-features", "id": "$USER/defia-features", "licenses": [{"name": "CC0-1.0"}]}
EOF
ds_ready() { kaggle datasets status "$USER/defia-features" 2>/dev/null | grep -qi ready; }
if ds_ready; then
    echo "[kaggle] defia-features existe -> nouvelle version"
    kaggle datasets version -p "$DSDIR" -m "update" --delete-old-versions || true
else
    echo "[kaggle] defia-features -> create"
    kaggle datasets create -p "$DSDIR" || true
fi
echo "[kaggle] attente dataset ready..."
for i in $(seq 1 40); do ds_ready && break; sleep 15; done
ds_ready || { echo "ERREUR dataset pas ready"; exit 1; }

# 2) Kernel GBM
cat > "$KDIR/kernel-metadata.json" <<EOF
{"id": "$USER/defia-gbm", "title": "defia-gbm", "code_file": "gbm_kernel.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": false, "enable_internet": true,
 "dataset_sources": ["$USER/defia-features"], "competition_sources": [], "kernel_sources": []}
EOF
echo "[kaggle] push kernel GBM..."
kaggle kernels push -p "$KDIR"

# 3) Poll + pull
KID="$USER/defia-gbm"
for i in $(seq 1 120); do
    ST=$(kaggle kernels status "$KID" 2>/dev/null | grep -oiE '(complete|error|running|queued|cancel)' | head -1 | tr '[:upper:]' '[:lower:]')
    echo "  [$i] $(date +%H:%M:%S) status=${ST:-unknown}"
    [ "$ST" = "complete" ] || [ "$ST" = "error" ] || [ "$ST" = "cancel" ] && break
    sleep 30
done
if [ "$ST" = "complete" ]; then
    kaggle kernels output "$KID" -p data/processed/oof 2>&1 | tail -3
    echo "[kaggle] sorties rapatriées."
fi
echo "PUSH_GBM_DONE status=$ST"
