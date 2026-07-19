#!/usr/bin/env bash
# Génération HORS ÉCHANTILLON des features issues du fine-tuning, par validation croisée.
#
# Problème corrigé : le modèle fine-tuné mémorise une partie de ses lignes d'entraînement. Ses
# features y paraissent bien meilleures qu'elles ne le sont (AUC 0,68 contre 0,61 sur données
# neuves pour emb_p1). Le GBM, entraîné sur ces mêmes lignes, leur fait donc trop confiance —
# d'où un best_iter effondré (2401 au lieu de 5360) et une MAE dégradée.
#
# Correctif : deux plis. Chaque moitié des lignes reçoit une prédiction faite par un modèle qui
# ne l'a jamais vue. Coût total = un entraînement complet, puisque chaque modèle voit la moitié.
set -u
cd "C:/Users/FLORIAN/Documents/Codes_Projes_Dev/Defi-IA-2020" || exit 1
S="C:/Users/FLORIAN/AppData/Local/Temp/claude/C--Users-FLORIAN-Documents-Codes-Projes-Dev-Defi-IA-2020/59329b65-af63-4d5d-bb8a-a1a86803dcd9/scratchpad"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN= HF_HUB_DISABLE_SYMLINKS_WARNING=1
P=data/processed
TAG=cf_buckets
MAX_RETRY=6

for fold in 0 1; do
  try=0
  while [ ! -f "reports/${TAG}_fold${fold}.json" ] && [ $try -lt $MAX_RETRY ]; do
    try=$((try + 1))
    echo "[cf] $(date '+%H:%M:%S') pli $fold (tentative $try/$MAX_RETRY)"
    .venv/Scripts/python.exe -u scripts/finetune.py --model intfloat/e5-small-v2 --tag "$TAG" \
      --objective buckets --batch 256 --epochs 2 --prefix "passage: " \
      --crossfit-k 2 --crossfit-fold "$fold" >> "$S/cf_fold${fold}.log" 2>&1
    [ -f "reports/${TAG}_fold${fold}.json" ] || { echo "[cf] interrompu, reprise dans 30s"; tail -3 "$S/cf_fold${fold}.log"; sleep 30; }
  done
  [ -f "reports/${TAG}_fold${fold}.json" ] || { echo "[cf] pli $fold ABANDONNE"; exit 1; }
  echo "[cf] $(date '+%H:%M:%S') pli $fold TERMINE"
done

echo "[cf] $(date '+%H:%M:%S') assemblage des plis"
.venv/Scripts/python.exe - <<'PY'
import pandas as pd
CF, TAG = "data/processed/crossfit", "cf_buckets"
# Lignes du train : concatener les deux moities, chacune predite par le modele qui ne l'a pas vue.
held = pd.concat([pd.read_parquet(f"{CF}/{TAG}_fold{k}_held.parquet") for k in (0, 1)])
# Holdout et test : moyenne des deux modeles (aucun des deux ne les a vus).
def mean_parts(part):
    a, b = (pd.read_parquet(f"{CF}/{TAG}_fold{k}_{part}.parquet").set_index("id") for k in (0, 1))
    return ((a + b.reindex(a.index)) / 2).reset_index()
val, test = mean_parts("val"), mean_parts("test")
train = pd.concat([held, val]).drop_duplicates("id")
for name, df in (("train", train), ("test", test)):
    base = pd.read_parquet(f"data/processed/{name}_emb.parquet")   # 64 figees + emb_p1 (hybride v1)
    m = base.merge(df, on="id", how="left")
    assert m.notna().all().all(), f"jointure incomplete sur {name}"
    m.to_parquet(f"data/processed/{name}_emb_cf.parquet", compression="zstd", index=False)
    print(f"{name}: {len(m):,} lignes, {m.shape[1]-1} features")
PY

restore() { for s in train test; do [ -f "$P/${s}_emb_ref.bak" ] && mv -f "$P/${s}_emb_ref.bak" "$P/${s}_emb.parquet"; done; echo "[cf] hybride v1 restaure"; }
trap restore EXIT
for s in train test; do
  cp -f "$P/${s}_emb.parquet" "$P/${s}_emb_ref.bak"
  cp -f "$P/${s}_emb_cf.parquet" "$P/${s}_emb.parquet"
done
echo "[cf] $(date '+%H:%M:%S') evaluation GBM"
.venv/Scripts/python.exe -u -m defia.cli train-gbm --eval-only --tag cf_hybrid --rounds 8000 \
  > "$S/eval_cf.log" 2>&1
grep -E "features, train|HOLDOUT|ATTENTION" "$S/eval_cf.log"
echo "[cf] references : hybride v1 = 7.9660 | v2 avec fuite = 8.0198 | record = 7.9077"
echo "[cf] ==== FIN $(date) ===="
