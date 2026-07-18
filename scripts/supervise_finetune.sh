#!/usr/bin/env bash
# Superviseur des fine-tunings : enchaîne e5 (~2h20) puis gte-modernbert (~12h45), avec
# relance automatique sur crash. Chaque relance repart du dernier checkpoint (finetune.py
# détecte models/<tag>/checkpoint.pt), donc une coupure coûte au pire quelques minutes.
#
# Sûr à relancer : si un run est déjà terminé, son checkpoint porte done_training et le script
# saute directement à l'inférence. Relancer le superviseur entier ne refait donc pas le travail.
set -u
cd "C:/Users/FLORIAN/Documents/Codes_Projes_Dev/Defi-IA-2020" || exit 1
S="C:/Users/FLORIAN/AppData/Local/Temp/claude/C--Users-FLORIAN-Documents-Codes-Projes-Dev-Defi-IA-2020/59329b65-af63-4d5d-bb8a-a1a86803dcd9/scratchpad"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN= HF_HUB_DISABLE_SYMLINKS_WARNING=1
MAX_RETRY=8

run() {  # $1=tag  $2=modele  $3=batch  $4=prefixe
  local tag=$1 model=$2 batch=$3 prefix=$4 try=0
  if [ -f "reports/$tag.json" ]; then
    echo "[sup] $tag deja termine (reports/$tag.json existe) - saute"
    return 0
  fi
  while [ $try -lt $MAX_RETRY ]; do
    try=$((try + 1))
    echo "[sup] $(date '+%H:%M:%S') lancement $tag (tentative $try/$MAX_RETRY)"
    .venv/Scripts/python.exe -u scripts/finetune.py \
      --model "$model" --tag "$tag" --batch "$batch" --epochs 2 --prefix "$prefix" \
      >> "$S/ft_$tag.log" 2>&1
    if [ -f "reports/$tag.json" ]; then
      echo "[sup] $(date '+%H:%M:%S') $tag TERMINE : $(grep -o '\"mae_holdout\":[^,]*' "reports/$tag.json")"
      return 0
    fi
    echo "[sup] $(date '+%H:%M:%S') $tag interrompu, reprise depuis checkpoint dans 30s"
    tail -3 "$S/ft_$tag.log"
    sleep 30
  done
  echo "[sup] $tag ABANDONNE apres $MAX_RETRY tentatives"
  return 1
}

echo "[sup] ==== DEBUT $(date) ===="
run ft_e5 "intfloat/e5-small-v2" 256 "passage: "
run ft_mb "Alibaba-NLP/gte-modernbert-base" 64 ""

# Blend final intégrant les prédictions fine-tunées (defia blend découvre les OOF tout seul)
echo "[sup] $(date '+%H:%M:%S') blend final avec les modeles fine-tunes"
.venv/Scripts/python.exe -u -m defia.cli blend --tag final_ft > "$S/blend_ft.log" 2>&1
grep -E "modeles|BLEND|MAE" "$S/blend_ft.log" | tail -12
echo "[sup] ==== FIN $(date) ===="
