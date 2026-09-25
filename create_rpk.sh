#!/usr/bin/env bash
# Package models/packerOut.zip into models/network.rpk for the IMX500 AI Camera.
#
# Official packaging must run on a Raspberry Pi:
#   sudo apt update
#   sudo apt install imx500-tools
#   ./create_rpk.sh
#
# Input (already in models/ from the IMX converter / YOLO IMX export):
#   packerOut.zip, dnnParams.xml, labels.txt, model_imx.onnx

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODELS="$ROOT/models"
ZIP="$MODELS/packerOut.zip"
OUT="$MODELS/rpk_out"

if [[ ! -f "$ZIP" ]]; then
  echo "Missing $ZIP"
  echo "Export/convert the model first so models/ contains packerOut.zip"
  exit 1
fi

if ! command -v imx500-package >/dev/null 2>&1; then
  echo "imx500-package was not found on this machine."
  echo
  echo "The converter already finished: models/packerOut.zip is ready."
  echo "Create the .rpk on a Raspberry Pi with:"
  echo "  sudo apt update && sudo apt install imx500-tools"
  echo "  cd $ROOT"
  echo "  ./create_rpk.sh"
  exit 2
fi

rm -rf "$OUT"
mkdir -p "$OUT"

echo "Packaging $ZIP -> $OUT"
imx500-package -i "$ZIP" -o "$OUT"

if [[ -f "$OUT/network.rpk" ]]; then
  cp -f "$OUT/network.rpk" "$MODELS/network.rpk"
  echo "Created $MODELS/network.rpk"
else
  echo "Packager ran but network.rpk was not found in $OUT"
  ls -la "$OUT"
  exit 3
fi
