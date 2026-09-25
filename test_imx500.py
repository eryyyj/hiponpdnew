#!/usr/bin/env python3
"""
Test the IMX500 camera with a known-good network (or a path you pass).

  python3 test_imx500.py
  python3 test_imx500.py --model /usr/share/imx500-models/some.rpk
  python3 test_imx500.py --model models/network.rpk
"""
import argparse
import glob
import os
import sys
import time

STOCK_DIR = "/usr/share/imx500-models"


def find_stock_rpk():
    preferred = [
        os.path.join(STOCK_DIR, "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"),
        os.path.join(STOCK_DIR, "imx500_network_yolov8n_pp.rpk"),
        os.path.join(STOCK_DIR, "imx500_network_yolo11n_pp.rpk"),
    ]
    for path in preferred:
        if os.path.isfile(path):
            return path
    found = sorted(glob.glob(os.path.join(STOCK_DIR, "*.rpk")))
    return found[0] if found else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to a .rpk network")
    parser.add_argument(
        "--no-nn",
        action="store_true",
        help="Preview only (no IMX500 network). Isolates CFE vs .rpk issues.",
    )
    args = parser.parse_args()

    from picamera2 import Picamera2

    if args.no_nn:
        print("Preview-only test (no .rpk / no on-sensor network)")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(buffer_count=2)
        print("Starting camera (no neural network)...")
        picam2.start(config, show_preview=False)
    else:
        model = args.model or find_stock_rpk()
        if not model:
            print("No .rpk found. Install stock models with:")
            print("  sudo apt install -y imx500-models")
            print(f"then: ls {STOCK_DIR}")
            print("Or test the sensor without a network:")
            print("  python3 test_imx500.py --no-nn")
            sys.exit(1)
        if not os.path.isfile(model):
            print(f"Missing model: {model}")
            sys.exit(1)

        print(f"Using model: {model} ({os.path.getsize(model)} bytes)")
        from picamera2.devices import IMX500

        imx500 = IMX500(model)
        picam2 = Picamera2(imx500.camera_num)
        config = picam2.create_preview_configuration(buffer_count=4)
        print("Starting camera with IMX500 network...")
        picam2.start(config, show_preview=False)

    deadline = time.time() + 180
    last = 0
    while time.time() < deadline:
        try:
            req = picam2.capture_request(wait=2.0)
            if req is None:
                raise TimeoutError("capture timed out")
            arr = req.make_array("main")
            req.release()
            print(f"SUCCESS: got a frame {arr.shape}")
            picam2.stop()
            return
        except Exception as exc:
            now = time.time()
            if now - last > 5:
                print(f"Still waiting for a frame: {exc}")
                last = now

    print("FAILED: no frame after 3 minutes")
    try:
        picam2.stop()
    except Exception:
        pass
    sys.exit(2)


if __name__ == "__main__":
    main()
