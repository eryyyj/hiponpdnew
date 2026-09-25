#!/usr/bin/env bash
# System-level fix for Pi 5 IMX500 CFE "Failed to queue buffer" errors.
# Installs IMX500 firmware/models and raises CMA memory, then asks for reboot.
set -euo pipefail

CMDLINE=""
for candidate in /boot/firmware/cmdline.txt /boot/cmdline.txt; do
  if [[ -f "$candidate" ]]; then
    CMDLINE="$candidate"
    break
  fi
done

CONFIG=""
for candidate in /boot/firmware/config.txt /boot/config.txt; do
  if [[ -f "$candidate" ]]; then
    CONFIG="$candidate"
    break
  fi
done

echo "==> Installing IMX500 firmware, models, and camera packages"
sudo apt-get update
sudo apt-get install -y \
  imx500-all \
  imx500-firmware \
  imx500-models \
  imx500-tools \
  python3-picamera2 \
  python3-libcamera \
  libcamera-apps \
  python3-numpy \
  python3-opencv || true

# imx500-all may not exist on every release; try firmware packages separately.
sudo apt-get install -y imx500-firmware imx500-models python3-picamera2 rpicam-apps || true

echo "==> Raising CMA memory to 512M (needed for IMX500 RAW + preview on Pi 5)"
if [[ -n "$CMDLINE" ]]; then
  sudo cp -a "$CMDLINE" "${CMDLINE}.bak.hipon"
  if grep -q 'cma=' "$CMDLINE"; then
    echo "cmdline already has a cma= setting:"
    grep -o 'cma=[^ ]*' "$CMDLINE" || true
  else
    # cmdline.txt must stay a single line
    sudo sed -i 's/$/ cma=512M/' "$CMDLINE"
    echo "Added cma=512M to $CMDLINE"
  fi
else
  echo "Could not find cmdline.txt"
fi

echo "==> Camera device permissions"
sudo usermod -aG video,render "$USER" || true
echo "Current groups for $USER: $(id -nG "$USER" 2>/dev/null || true)"

echo "==> Stopping desktop camera consumers (PipeWire) that hold /dev/video*"
systemctl --user stop pipewire.socket pipewire pipewire-pulse wireplumber 2>/dev/null || true
pkill -9 -f wireplumber 2>/dev/null || true
pkill -9 -f pipewire 2>/dev/null || true
pkill -9 -f rpicam 2>/dev/null || true
pkill -9 -f libcamera 2>/dev/null || true

if [[ -n "$CONFIG" ]]; then
  if ! grep -qE '^camera_auto_detect=1' "$CONFIG"; then
    echo "Adding camera_auto_detect=1 to $CONFIG"
    echo "camera_auto_detect=1" | sudo tee -a "$CONFIG" >/dev/null
  fi
fi

echo
echo "Packages and CMA are set. You MUST reboot for CMA to apply:"
echo "  sudo reboot"
echo
echo "After reboot, log out/in if you were added to the video group, then:"
echo "  rpicam-hello -t 5000"
echo "  python3 test_imx500.py --no-nn"
echo "  python3 test_imx500.py --model models/network.rpk"
