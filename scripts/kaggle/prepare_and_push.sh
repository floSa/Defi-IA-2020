#!/usr/bin/env bash
# Orchestration Kaggle : export data -> push dataset -> push kernel(s) -> poll -> pull résultats.
#
# Prérequis : ~/.kaggle/kaggle.json présent (chmod 600) et `kaggle` installé dans le venv.
# Usage : bash scripts/kaggle/prepare_and_push.sh embeddings|transformer
set -eo pipefail
cd "$(dirname "$0")/../.."
# activate peut référencer des variables non-définies (LD_LIBRARY_PATH) -> on désactive nounset le temps du source
set +u; . .venv/bin/activate; set -u

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

# 1+2) Dataset Kaggle : on saute tout si déjà "ready" (données immuables). Sinon export + upload.
#      Le CLI Kaggle peut renvoyer une erreur JSON de finalisation ("Expecting value...") alors
#      que l'upload a RÉUSSI -> on tolère (|| true) et on vérifie le statut réel via `status`.
dataset_ready() { kaggle datasets status "$USERNAME/$DATASET_SLUG" 2>/dev/null | grep -qi ready; }

if dataset_ready; then
    echo "[kaggle] dataset $USERNAME/$DATASET_SLUG déjà prêt -> upload sauté"
else
    if [ ! -f "$DATASET_DIR/train.parquet" ]; then
        echo "[kaggle] export des données..."
        PYTHONPATH=src python -m defia.cli kaggle-export --config configs/default.yaml --out "$DATASET_DIR"
    fi
    cat > "$DATASET_DIR/dataset-metadata.json" <<EOF
{"title": "$DATASET_SLUG", "id": "$USERNAME/$DATASET_SLUG", "licenses": [{"name": "CC0-1.0"}]}
EOF
    if kaggle datasets list --user "$USERNAME" 2>/dev/null | grep -q "$DATASET_SLUG"; then
        echo "[kaggle] dataset existe -> nouvelle version"
        kaggle datasets version -p "$DATASET_DIR" -m "update" --delete-old-versions || true
    else
        echo "[kaggle] dataset absent -> create"
        kaggle datasets create -p "$DATASET_DIR" || true
    fi
    echo "[kaggle] attente statut 'ready'..."
    for i in $(seq 1 40); do dataset_ready && break; sleep 15; done
    dataset_ready || { echo "ERREUR: dataset jamais 'ready'." >&2; exit 1; }
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
# Le CLI renvoie: ... has status "KernelWorkerStatus.RUNNING|COMPLETE|ERROR|QUEUED|..."
for i in $(seq 1 240); do   # jusqu'à ~2h (30s * 240)
    RAW="$(kaggle kernels status "$KERNEL_ID" 2>/dev/null || true)"
    STATUS="$(echo "$RAW" | grep -oiE '(complete|error|running|queued|cancel)' | head -1 | tr '[:upper:]' '[:lower:]')"
    STATUS="${STATUS:-unknown}"
    echo "  [$i] status=$STATUS"
    if [ "$STATUS" = "complete" ] || [ "$STATUS" = "error" ] || [ "$STATUS" = "cancel" ]; then
        break
    fi
    sleep 30
done

if [ "$STATUS" != "complete" ]; then
    echo "ERREUR: kernel terminé avec status=$STATUS (voir https://www.kaggle.com/code/$KERNEL_ID)." >&2
    exit 1
fi

# 5) Rapatrie les sorties
OUT_DIR="data/processed"
mkdir -p "$OUT_DIR"
echo "[kaggle] rapatriement des sorties -> $OUT_DIR"
kaggle kernels output "$KERNEL_ID" -p "$OUT_DIR"
echo "[kaggle] terminé. Fichiers dans $OUT_DIR"
