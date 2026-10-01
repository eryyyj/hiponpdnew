# Shrimp Counting App for Raspberry Pi 5

This project runs a Flask web app for shrimp counting on a Raspberry Pi, with the new NCNN model path preferred and the original IMX500 path kept as a fallback.

## Overview

The app does the following:

- Starts a Flask UI for camera/control/automation
- Uses a camera feed for live shrimp detection
- Draws bounding boxes and a count line over the image
- Counts shrimp as they cross the detection line
- Exposes the app on a local web page and optionally kiosk mode

The app is configured to prefer the NCNN model on Raspberry Pi 5 when the model files are present. If the NCNN export is missing, it falls back to the IMX500 flow.

## Hardware

- Raspberry Pi 5 (8 GB recommended)
- Raspberry Pi Camera Module or compatible USB camera
- Optional: Raspberry Pi AI Camera / IMX500 hardware if you want to use the original IMX500 path
- ESP32 serial device if you use the feeder/automation features

## Operating system

Use Raspberry Pi OS (64-bit) with desktop environment.

Recommended:

```bash
sudo apt update
sudo apt upgrade -y
```

## Install system packages

Run the following:

```bash
sudo apt install -y \
  python3 python3-pip python3-venv python3-dev \
  python3-pil python3-opencv python3-picamera2 \
  python3-numpy chromium-browser libcap-dev \
  python3-serial
```

If `python3-picamera2` is not available in your image, install the latest Raspberry Pi OS build that includes libcamera and picamera2 support.

## Create a Python virtual environment

From the project folder:

```bash
cd ~/hiponpdnew
python3 -m venv .venv
source .venv/bin/activate
```

## Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `pip install ncnn` fails on your Pi, try the distro package if available or build from source for your Pi architecture.

## Prepare the NCNN model files

This app expects the NCNN model export in the `models/` folder, matching the exact Ultralytics export structure from:

```bash
yolo export model=best.pt format=ncnn imgsz=640
```

The expected layout is:

```bash
models/
  best_ncnn_model/
    model.ncnn.param
    model.ncnn.bin
    labels.txt   # if exported with labels
```

Legacy names like `shrimp_ncnn.param` / `shrimp_ncnn.bin` are still accepted for compatibility, but the default export path matches the Ultralytics output exactly. If your files have different names, update these constants in `main.py`:

```python
NCNN_MODEL_PARAM = os.path.join(..., "models", "best_ncnn_model", "model.ncnn.param")
NCNN_MODEL_BIN = os.path.join(..., "models", "best_ncnn_model", "model.ncnn.bin")
NCNN_LABELS_PATH = os.path.join(..., "models", "best_ncnn_model", "labels.txt")
```

For the IMX500 fallback, the old files are also referenced in `main.py`:

```python
IMX500_MODEL_PATH = "/home/admin/Desktop/hipon/models/network.rpk"
IMX500_LABELS_PATH = "/home/admin/Desktop/hipon/models/labels.txt"
```

## Run the app

From the project folder:

```bash
cd ~/hiponpdnew
source .venv/bin/activate
python3 main.py
```

The app will:

- initialize the camera pipeline
- prefer the NCNN model on Pi 5 when present
- fall back to IMX500 if needed
- serve the web interface on port 5000

Open the app in a browser at:

```text
http://<pi-ip>:5000
```

or locally:

```text
http://localhost:5000
```

## Kiosk mode

On Raspberry Pi OS desktop, the app tries to launch Chromium in kiosk mode automatically. If a browser is not available, it will print the URL and continue running in terminal mode.

## Troubleshooting

### NCNN model not loading

Check that the files exist:

```bash
ls -l models
```

Verify the names match the constants in `main.py`.

### Camera not opening

Try:

```bash
ls /dev/video*
```

If no camera is detected, confirm the camera is enabled in Raspberry Pi configuration.

### App fails to start

Check the installed packages:

```bash
pip list | grep -E "flask|numpy|opencv|ncnn|picamera2"
```

Also verify Python is using the virtual environment:

```bash
which python
python --version
```

## Useful notes

- The app currently prefers the NCNN detection path for the Pi 5 because it works with a standard camera input and 640x640 preprocessing.
- The IMX500 code path remains in place as a fallback for the original Raspberry Pi AI Camera workflow.
- This project includes a browser UI, serial control hooks, and automation logic for the feeder. If you are only testing detection, you can run the detection code in a simpler script and ignore the full app UI.

## Default app entry point

```bash
python3 main.py
```

## Optional: run in background

```bash
nohup python3 main.py > shrimp_app.log 2>&1 &
```

Then inspect the log:

```bash
tail -f shrimp_app.log
```
