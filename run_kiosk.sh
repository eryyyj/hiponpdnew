#!/usr/bin/env bash
# Launches the Hipon app from the project venv. main.py starts Flask
# and opens the kiosk browser.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export DISPLAY="${DISPLAY:-:0}"

VENV="$ROOT/.venv"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "No venv at $VENV — run ./install.sh first."
  exit 1
fi

exec python3 "$ROOT/main.py"
