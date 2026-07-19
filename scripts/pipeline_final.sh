#!/usr/bin/env bash
# Pipeline aval complet sur la base entièrement hors échantillon (70 features : 64 embeddings
# figés + 5 probabilités de tranches + espérance d'upvotes, ces 6 dernières générées en
# validation croisée à 2 plis).
#
# Base : MAE 7.9387 en modèle simple. Le pipeline aval valait -0,058 la fois précédente.
set -u
cd "C:/Users/FLORIAN/Documents/Codes_Projes_Dev/Defi-IA-2020" || exit 1
S="C:/Users/FLORIAN/AppData/Local/Temp/claude/C--Users-FLORIAN-Documents-Codes-Projes-Dev-Defi-IA-2020/59329b65-af63-4d5d-bb8a-a1a86803dcd9/scratchpad"
P=data/processed

# Les features propres deviennent la référence ; l'hybride v1 (fuyant) est conservé à côté.
[ -f "$P/train_emb_hybrid1.parquet" ] || for s in train test; do
  cp -f "$P/${s}_emb.parquet" "$P/${s}_emb_hybrid1.parquet"
done
for s in train test; do cp -f "$P/${s}_emb_cfclean.parquet" "$P/${s}_emb.parquet"; done
echo "[p] base = croisee sans fuite (70 features)"

echo "[p] $(date '+%H:%M:%S') --- 1/4 GBM : soumission + OOF ---"
.venv/Scripts/python.exe -u -m defia.cli train-gbm --tag clean_full --rounds 8000 \
  > "$S/p_gbm.log" 2>&1
grep -E "HOLDOUT|ATTENTION|submission" "$S/p_gbm.log"

echo "[p] $(date '+%H:%M:%S') --- 2/4 deux etages ---"
sed -i 's|reports/gbm_hybrid_full.json|reports/gbm_clean_full.json|' scripts/two_stage.py
.venv/Scripts/python.exe -u scripts/two_stage.py > "$S/p_2stage.log" 2>&1
grep -E "etage A|MAE sur|holdout entier|seuil retenu" "$S/p_2stage.log"
.venv/Scripts/python.exe -u scripts/two_stage_submit.py > "$S/p_2sub.log" 2>&1
grep -E "MAE holdout|crit " "$S/p_2sub.log"

echo "[p] $(date '+%H:%M:%S') --- 3/4 blend ---"
.venv/Scripts/python.exe -u -m defia.cli blend --tag final_clean > "$S/p_blend.log" 2>&1
grep -E "poids|BLEND|soumission" "$S/p_blend.log"

echo "[p] $(date '+%H:%M:%S') --- 4/4 regle deux etages sur le blend ---"
sed -i 's|blend_final_hybrid.json|blend_final_clean.json|' scripts/blend_two_stage.py
.venv/Scripts/python.exe -u scripts/blend_two_stage.py > "$S/p_blend2s.log" 2>&1
grep -E "seuil retenu|2e moiti|holdout entier" "$S/p_blend2s.log"

sed -i 's|submission_final_hybrid.csv|submission_final_clean.csv|; s|submission_final_hybrid2s.csv|submission_final_clean2s.csv|' scripts/blend_two_stage_submit.py
.venv/Scripts/python.exe -u scripts/blend_two_stage_submit.py > "$S/p_finalsub.log" 2>&1
grep -E "forcees|crit " "$S/p_finalsub.log"
echo "[p] ==== FIN $(date) ===="
