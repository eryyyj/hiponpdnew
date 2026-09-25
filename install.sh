#!/usr/bin/env bash
# One-time setup for Raspberry Pi:
#   - create and activate a Python venv, then install requirements.txt
#   - package the IMX500 model (create_rpk.sh)
#   - create a Desktop shortcut with landing.png as the icon
# The shortcut runs the app in kiosk mode (handled by main.py).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ICON_SRC="$ROOT/assets/landing.png"
ICON_SPLASH_DIR="$ROOT/assets/images"
DESKTOP_DIR="${HOME}/Desktop"
APPS_DIR="${HOME}/.local/share/applications"
SHORTCUT_NAME="Shrimp Farm Control"
DESKTOP_FILE="$DESKTOP_DIR/hipon.desktop"
APPS_FILE="$APPS_DIR/hipon.desktop"

echo "==> Project: $ROOT"

if [[ ! -f "$ICON_SRC" ]]; then
  echo "Missing icon: $ICON_SRC"
  exit 1
fi

mkdir -p "$ICON_SPLASH_DIR"
cp -f "$ICON_SRC" "$ICON_SPLASH_DIR/landing.png"

echo "==> Installing system packages"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-pil \
    python3-numpy \
    python3-opencv \
    python3-picamera2 \
    python3-libcamera \
    libcamera-apps \
    libcap-dev \
    chromium-browser \
    imx500-tools || true
  if ! command -v chromium-browser >/dev/null 2>&1 && ! command -v chromium >/dev/null 2>&1; then
    sudo apt-get install -y chromium || true
  fi
else
  echo "apt-get not found; skipping system packages (install them manually)."
fi

echo "==> Creating virtual environment"
VENV="$ROOT/.venv"
if [[ ! -d "$VENV" ]]; then
  # --system-site-packages lets the venv use apt-installed picamera2/PIL on the Pi.
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python3 -m pip install --upgrade pip
# Use apt builds of these so numpy ABI matches picamera2/simplejpeg.
python3 -m pip uninstall -y picamera2 numpy opencv-python opencv-python-headless >/dev/null 2>&1 || true

echo "==> Installing Python requirements into the venv"
if ! python3 -m pip install -r "$ROOT/requirements.txt"; then
  echo "Full pip install failed; installing packages except picamera2"
  grep -viE '^[[:space:]]*picamera2[[:space:]]*$|^[[:space:]]*#|^[[:space:]]*$' \
    "$ROOT/requirements.txt" > /tmp/hipon-requirements.txt
  python3 -m pip install -r /tmp/hipon-requirements.txt
fi

echo "==> Creating IMX500 RPK"
chmod +x "$ROOT/create_rpk.sh" "$ROOT/run_kiosk.sh"
if ! "$ROOT/create_rpk.sh"; then
  echo "Warning: create_rpk.sh did not finish. Install imx500-tools on the Pi and re-run:"
  echo "  $ROOT/create_rpk.sh"
fi

echo "==> Creating desktop shortcut"
mkdir -p "$DESKTOP_DIR" "$APPS_DIR"
chmod +x "$ROOT/run_kiosk.sh"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$SHORTCUT_NAME
Comment=Launch Shrimp Farm Control in kiosk mode
Exec=$ROOT/run_kiosk.sh
Path=$ROOT
Icon=$ICON_SRC
Terminal=false
Categories=Utility;
StartupNotify=true
EOF

cp -f "$DESKTOP_FILE" "$APPS_FILE"
chmod +x "$DESKTOP_FILE" "$APPS_FILE"

if command -v gio >/dev/null 2>&1; then
  gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
fi

echo
echo "Done."
echo "Python venv: $VENV"
echo "Desktop shortcut: $DESKTOP_FILE"
echo "Icon: $ICON_SRC"
echo "Double-click \"$SHORTCUT_NAME\" to start kiosk mode."
