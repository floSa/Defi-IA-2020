#!/usr/bin/env bash
# Contourne l'absence de libgomp.so.1 système (LightGBM en a besoin) sans sudo :
# on réutilise la copie vendue par scikit-learn et on l'expose via LD_LIBRARY_PATH.
set -e
cd "$(dirname "$0")/.."
. .venv/bin/activate
LIBS_DIR="$(python -c 'import os,glob,sklearn; b=os.path.dirname(os.path.dirname(sklearn.__file__)); print(glob.glob(os.path.join(b,"scikit_learn.libs"))[0])')"
GOMP="$(ls "$LIBS_DIR"/libgomp-*.so.* | head -1)"
ln -sf "$GOMP" "$LIBS_DIR/libgomp.so.1"
echo "symlink: $LIBS_DIR/libgomp.so.1 -> $GOMP"

# Rend LD_LIBRARY_PATH persistant à l'activation du venv (idempotent)
ACT=.venv/bin/activate
LINE="export LD_LIBRARY_PATH=\"$LIBS_DIR:\${LD_LIBRARY_PATH:-}\""
grep -qF "$LIBS_DIR" "$ACT" || echo "$LINE" >> "$ACT"
echo "activate patché."
