#!/usr/bin/env bash
# Fine-tuning avec l'objectif CORRIGÉ (classification ups==1), puis évaluation GBM immédiate.
#
# Pourquoi ce changement : le premier fine-tuning, en régression L1 directe sur ups, s'est
# effondré sur la médiane (MAE 11.903 = la baseline « prédire 1 partout »). C'est mécanique :
# 52 % des ups valent exactement 1, donc la constante optimale en L1 EST la médiane, et la
# descente de gradient s'y gare. La classification ups==1 est équilibrée 52/48 et n'a pas ce
# point fixe dégénéré.
#
# Relance automatique sur crash (reprise depuis checkpoint). Sûr à relancer entièrement.
set -u
cd "C:/Users/FLORIAN/Documents/Codes_Projes_Dev/Defi-IA-2020" || exit 1
S="C:/Users/FLORIAN/AppData/Local/Temp/claude/C--Users-FLORIAN-Documents-Codes-Projes-Dev-Defi-IA-2020/59329b65-af63-4d5d-bb8a-a1a86803dcd9/scratchpad"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN= HF_HUB_DISABLE_SYMLINKS_WARNING=1
P=data/processed
TAG=ft_e5_cls
MAX_RETRY=8

try=0
while [ ! -f "reports/$TAG.json" ] && [ $try -lt $MAX_RETRY ]; do
  try=$((try + 1))
  echo "[sup] $(date '+%H:%M:%S') fine-tuning $TAG (tentative $try/$MAX_RETRY)"
  .venv/Scripts/python.exe -u scripts/finetune.py --model intfloat/e5-small-v2 --tag "$TAG" \
    --objective cls --batch 256 --epochs 2 --prefix "passage: " >> "$S/ft_$TAG.log" 2>&1
  [ -f "reports/$TAG.json" ] || { echo "[sup] interrompu, reprise dans 30s"; tail -3 "$S/ft_$TAG.log"; sleep 30; }
done
[ -f "reports/$TAG.json" ] || { echo "[sup] $TAG ABANDONNE"; exit 1; }
echo "[sup] $(date '+%H:%M:%S') $TAG TERMINE : $(grep -o '\"auc_holdout\":[^,]*' "reports/$TAG.json")"

# --- Évaluation : ces embeddings + la proba texte valent-ils mieux que les embeddings figés ? ---
restore() { for s in train test; do [ -f "$P/${s}_emb_ref.bak" ] && mv -f "$P/${s}_emb_ref.bak" "$P/${s}_emb.parquet"; done; echo "[sup] embeddings figes restaures"; }
trap restore EXIT
for s in train test; do
  cp -f "$P/${s}_emb.parquet" "$P/${s}_emb_ref.bak"
  cp -f "$P/${s}_${TAG}emb.parquet" "$P/${s}_emb.parquet"
done
echo "[sup] $(date '+%H:%M:%S') evaluation GBM avec les embeddings fine-tunes"
.venv/Scripts/python.exe -u -m defia.cli train-gbm --eval-only --tag gbm_${TAG} --rounds 6000 \
  > "$S/eval_$TAG.log" 2>&1
grep -E "features, train|HOLDOUT|ATTENTION" "$S/eval_$TAG.log"
echo "[sup] reference embeddings figes = 7.9968"
echo "[sup] ==== FIN $(date) ===="
