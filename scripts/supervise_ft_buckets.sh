#!/usr/bin/env bash
# Fine-tuning à objectif "tranches d'upvotes" ({1} · 2-3 · 4-10 · 11-50 · 51+), puis évaluation.
#
# Pourquoi ces tranches : le binaire ups==1 n'apprend au modèle qu'à repérer le pic. Les tranches
# lui font apprendre l'INTENSITÉ de la viralité, qui est là où se joue la MAE (la queue). Smoke
# test : AUC 0.6074 sur 80k lignes / 1 époque contre 0.6057 pour le binaire sur 2,28M / 2 époques.
#
# Relance automatique sur crash (reprise depuis checkpoint). Sûr à relancer entièrement.
set -u
cd "C:/Users/FLORIAN/Documents/Codes_Projes_Dev/Defi-IA-2020" || exit 1
S="C:/Users/FLORIAN/AppData/Local/Temp/claude/C--Users-FLORIAN-Documents-Codes-Projes-Dev-Defi-IA-2020/59329b65-af63-4d5d-bb8a-a1a86803dcd9/scratchpad"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN= HF_HUB_DISABLE_SYMLINKS_WARNING=1
P=data/processed
TAG=ft_e5_buckets
MAX_RETRY=8

try=0
while [ ! -f "reports/$TAG.json" ] && [ $try -lt $MAX_RETRY ]; do
  try=$((try + 1))
  echo "[sup] $(date '+%H:%M:%S') fine-tuning $TAG (tentative $try/$MAX_RETRY)"
  .venv/Scripts/python.exe -u scripts/finetune.py --model intfloat/e5-small-v2 --tag "$TAG" \
    --objective buckets --batch 256 --epochs 2 --prefix "passage: " >> "$S/ft_$TAG.log" 2>&1
  [ -f "reports/$TAG.json" ] || { echo "[sup] interrompu, reprise dans 30s"; tail -3 "$S/ft_$TAG.log"; sleep 30; }
done
[ -f "reports/$TAG.json" ] || { echo "[sup] $TAG ABANDONNE"; exit 1; }
echo "[sup] $(date '+%H:%M:%S') $TAG TERMINE : $(grep -o '\"auc_holdout\":[^,]*' "reports/$TAG.json")"

# --- Hybride v2 : embeddings figés + proba binaire + les 5 probas de tranches + l'espérance ---
echo "[sup] $(date '+%H:%M:%S') construction de l'hybride v2"
.venv/Scripts/python.exe - <<'PY'
import pandas as pd
for s in ("train", "test"):
    base = pd.read_parquet(f"data/processed/{s}_emb_hybrid.parquet")   # 64 figées + emb_p1
    bk = pd.read_parquet(f"data/processed/{s}_ft_e5_bucketsemb.parquet")
    keep = ["id"] + [c for c in bk.columns if c.startswith(("emb_pb", "emb_exp"))]
    m = base.merge(bk[keep], on="id", how="left")
    assert m.notna().all().all(), f"jointure incomplete sur {s}"
    m.to_parquet(f"data/processed/{s}_emb_hybrid2.parquet", compression="zstd", index=False)
    print(f"{s}: {m.shape[1]-1} features")
PY

restore() { for s in train test; do [ -f "$P/${s}_emb_ref.bak" ] && mv -f "$P/${s}_emb_ref.bak" "$P/${s}_emb.parquet"; done; echo "[sup] hybride v1 restaure"; }
trap restore EXIT
for s in train test; do
  cp -f "$P/${s}_emb.parquet" "$P/${s}_emb_ref.bak"
  cp -f "$P/${s}_emb_hybrid2.parquet" "$P/${s}_emb.parquet"
done
echo "[sup] $(date '+%H:%M:%S') evaluation GBM de l'hybride v2"
.venv/Scripts/python.exe -u -m defia.cli train-gbm --eval-only --tag hybrid2 --rounds 8000 \
  > "$S/eval_hybrid2.log" 2>&1
grep -E "features, train|HOLDOUT|ATTENTION" "$S/eval_hybrid2.log"
echo "[sup] reference hybride v1 = 7.9660 | record actuel (blend+regle) = 7.9077"
echo "[sup] ==== FIN $(date) ===="
