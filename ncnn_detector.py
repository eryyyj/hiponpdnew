#!/usr/bin/env python3
"""
ncnn_detector.py - YOLO11 (Ultralytics ncnn export) shrimp detection + counting.

Replaces the IMX500 (.rpk / on-sensor) inference path. The AI Camera is now
used as a plain camera: Picamera2 captures frames, this module runs the ncnn
model on the Pi CPU, then tracks centroids and counts crossings of the
horizontal counting line (same tracker constants as main.py).

Export the model on your PC (not on the Pi):
    pip install ultralytics
    yolo export model=best.pt format=ncnn imgsz=640
-> best_ncnn_model/model.ncnn.param + model.ncnn.bin (+ metadata.yaml)

Pipeline per frame:
    frame (e.g. 640x480) -> letterbox to 640x640 -> RGB -> /255 -> ncnn
    -> (4+nc, 8400) tensor -> conf filter -> NMS -> map back to frame coords
    -> centroid tracker -> line-crossing count
"""

import threading
import time

import cv2
import numpy as np
import ncnn

# ---------------------------------------------------------------------------
# Config (mirrors the constants in main.py where they exist)
# ---------------------------------------------------------------------------
NCNN_PARAM_PATH = "/home/admin/Desktop/hipon/models/best_ncnn_model/model.ncnn.param"
NCNN_BIN_PATH = "/home/admin/Desktop/hipon/models/best_ncnn_model/model.ncnn.bin"
NCNN_INPUT_SIZE = 640

# Ultralytics' ncnn export names the blobs "in0" / "out0".
# If you exported differently, check the first and last lines of model.ncnn.param.
NCNN_INPUT_BLOB = "in0"
NCNN_OUTPUT_BLOB = "out0"

DETECTION_THRESHOLD = 0.55
DETECTION_IOU = 0.65
DETECTION_MAX_DETECTIONS = 10

MAX_TRACK_DISTANCE = 80        # px, centroid match distance between frames
MAX_DISAPPEARED_FRAMES = 100   # frames an object can be unseen before dropped

NUM_THREADS = 4                # Pi 4/5 have 4 cores
LETTERBOX_PAD_VALUE = 114      # Ultralytics' default padding gray


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class NcnnDetector:
    """Loads the ncnn model once; call detect(frame) for every frame."""

    def __init__(
        self,
        param_path=NCNN_PARAM_PATH,
        bin_path=NCNN_BIN_PATH,
        input_size=NCNN_INPUT_SIZE,
        conf_threshold=DETECTION_THRESHOLD,
        iou_threshold=DETECTION_IOU,
        max_detections=DETECTION_MAX_DETECTIONS,
        num_threads=NUM_THREADS,
        frame_is_bgr=False,
    ):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        # Set True if your frames are BGR-ordered in memory (cv2.imread style).
        # NOTE: Picamera2's "RGB888" format is actually laid out B,G,R in the
        # numpy array. If boxes look right but confidence is poor, flip this.
        self.frame_is_bgr = frame_is_bgr

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False   # no usable Vulkan on Pi CPU path
        self.net.opt.num_threads = num_threads
        if self.net.load_param(param_path) != 0:
            raise RuntimeError(f"Failed to load ncnn param: {param_path}")
        if self.net.load_model(bin_path) != 0:
            raise RuntimeError(f"Failed to load ncnn bin: {bin_path}")

    # -- preprocessing ------------------------------------------------------
    def preprocess(self, frame):
        """Letterbox to input_size x input_size, convert to an RGB ncnn.Mat.

        Returns (mat, scale, pad_left, pad_top) so boxes can be mapped back.
        Aspect ratio is preserved: a 640x480 frame becomes 640x480 content
        with 80 px gray bars top and bottom (scale = 1.0).
        """
        size = self.input_size
        h, w = frame.shape[:2]
        scale = min(size / h, size / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        resized = frame
        if (new_w, new_h) != (w, h):
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w, pad_h = size - new_w, size - new_h
        left, top = pad_w // 2, pad_h // 2
        padded = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT,
            value=(LETTERBOX_PAD_VALUE,) * 3,
        )

        if self.frame_is_bgr:
            padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = np.ascontiguousarray(padded, dtype=np.uint8)

        mat = ncnn.Mat.from_pixels(
            padded, ncnn.Mat.PixelType.PIXEL_RGB, size, size
        )
        mat.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0] * 3)
        return mat, scale, left, top

    # -- inference ----------------------------------------------------------
    def detect(self, frame):
        """Run the model. Returns a list of (x1, y1, x2, y2, conf, class_id)
        in ORIGINAL frame pixel coordinates."""
        mat, scale, pad_left, pad_top = self.preprocess(frame)

        with self.net.create_extractor() as ex:
            ex.input(NCNN_INPUT_BLOB, mat)
            ret, out = ex.extract(NCNN_OUTPUT_BLOB)
        if ret != 0:
            raise RuntimeError(f"ncnn extract('{NCNN_OUTPUT_BLOB}') failed: {ret}")

        pred = np.array(out)
        return self._postprocess(pred, frame.shape[:2], scale, pad_left, pad_top)

    # -- postprocessing -----------------------------------------------------
    def _postprocess(self, pred, frame_hw, scale, pad_left, pad_top):
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            raise ValueError(f"Unexpected ncnn output shape: {pred.shape}")
        # Ultralytics output is (4+nc, num_anchors); make it (num_anchors, 4+nc)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_cxcywh = pred[:, :4]
        class_scores = pred[:, 4:]            # already sigmoid-ed by the export
        class_ids = class_scores.argmax(axis=1)
        confs = class_scores.max(axis=1)

        keep = confs >= self.conf_threshold
        if not np.any(keep):
            return []
        boxes_cxcywh, confs, class_ids = boxes_cxcywh[keep], confs[keep], class_ids[keep]

        # letterbox space -> original frame space
        cx = (boxes_cxcywh[:, 0] - pad_left) / scale
        cy = (boxes_cxcywh[:, 1] - pad_top) / scale
        bw = boxes_cxcywh[:, 2] / scale
        bh = boxes_cxcywh[:, 3] / scale
        x1, y1 = cx - bw / 2, cy - bh / 2

        h, w = frame_hw
        nms_boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(
            nms_boxes, confs.tolist(), self.conf_threshold, self.iou_threshold
        )
        idxs = np.array(idxs).flatten()[: self.max_detections]

        dets = []
        for i in idxs:
            bx1 = float(np.clip(x1[i], 0, w - 1))
            by1 = float(np.clip(y1[i], 0, h - 1))
            bx2 = float(np.clip(x1[i] + bw[i], 0, w - 1))
            by2 = float(np.clip(y1[i] + bh[i], 0, h - 1))
            dets.append((bx1, by1, bx2, by2, float(confs[i]), int(class_ids[i])))
        return dets


# ---------------------------------------------------------------------------
# Centroid tracker + counting line
# ---------------------------------------------------------------------------
class CentroidTracker:
    def __init__(self, max_distance=MAX_TRACK_DISTANCE, max_disappeared=MAX_DISAPPEARED_FRAMES):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.next_id = 0
        self.objects = {}      # id -> (cx, cy)
        self.prev_y = {}       # id -> previous cy (for crossing test)
        self.disappeared = {}  # id -> frames unseen

    def update(self, centroids):
        """Greedy nearest-neighbour match. Returns {id: (cx, cy, prev_cy)}."""
        if not centroids:
            for oid in list(self.objects):
                self._mark_missing(oid)
            return {}

        if not self.objects:
            for c in centroids:
                self._register(c)
        else:
            ids = list(self.objects)
            obj_pts = np.array([self.objects[i] for i in ids], dtype=float)
            new_pts = np.array(centroids, dtype=float)
            dist = np.linalg.norm(obj_pts[:, None, :] - new_pts[None, :, :], axis=2)

            used_rows, used_cols = set(), set()
            for flat in np.argsort(dist, axis=None):
                r, c = divmod(int(flat), dist.shape[1])
                if r in used_rows or c in used_cols:
                    continue
                if dist[r, c] > self.max_distance:
                    break  # sorted ascending: everything after is farther
                oid = ids[r]
                self.prev_y[oid] = self.objects[oid][1]
                self.objects[oid] = tuple(centroids[c])
                self.disappeared[oid] = 0
                used_rows.add(r)
                used_cols.add(c)

            for r, oid in enumerate(ids):
                if r not in used_rows:
                    self._mark_missing(oid)
            for c, pt in enumerate(centroids):
                if c not in used_cols:
                    self._register(pt)

        return {
            oid: (pt[0], pt[1], self.prev_y.get(oid, pt[1]))
            for oid, pt in self.objects.items()
            if self.disappeared.get(oid, 0) == 0
        }

    def _register(self, pt):
        self.objects[self.next_id] = tuple(pt)
        self.prev_y[self.next_id] = pt[1]
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _mark_missing(self, oid):
        self.disappeared[oid] = self.disappeared.get(oid, 0) + 1
        if self.disappeared[oid] > self.max_disappeared:
            for d in (self.objects, self.prev_y, self.disappeared):
                d.pop(oid, None)

    def reset(self):
        self.__init__(self.max_distance, self.max_disappeared)


class ShrimpCounter:
    """Detector + tracker + line-crossing count. Thread-safe getters.

    direction: "down" counts crossings of the line top->bottom (y increasing),
               "up" counts bottom->top, "both" counts either.
    """

    def __init__(self, detector=None, line_y_fraction=0.80, direction="down"):
        self.detector = detector or NcnnDetector()
        self.tracker = CentroidTracker()
        self.line_y_fraction = line_y_fraction
        self.direction = direction
        self._counted = set()
        self._count = 0
        self._detections = []
        self._lock = threading.Lock()

    # same role as /api/count_line POST in main.py
    def set_line_fraction(self, fraction):
        with self._lock:
            self.line_y_fraction = float(fraction)

    @property
    def count(self):
        with self._lock:
            return self._count

    @property
    def detections(self):
        with self._lock:
            return list(self._detections)

    def reset_count(self):
        with self._lock:
            self._count = 0
            self._counted.clear()
            self.tracker.reset()

    def process(self, frame):
        """Run detection + tracking on one frame. Returns detections list."""
        dets = self.detector.detect(frame)
        centroids = [((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in dets]
        tracked = self.tracker.update(centroids)

        line_y = self.line_y_fraction * frame.shape[0]
        with self._lock:
            for oid, (_, cy, prev_cy) in tracked.items():
                if oid in self._counted:
                    continue
                crossed_down = prev_cy < line_y <= cy
                crossed_up = prev_cy > line_y >= cy
                if (
                    (self.direction == "down" and crossed_down)
                    or (self.direction == "up" and crossed_up)
                    or (self.direction == "both" and (crossed_down or crossed_up))
                ):
                    self._counted.add(oid)
                    self._count += 1
            self._detections = dets
        return dets


# ---------------------------------------------------------------------------
# Background inference worker (keeps Flask / the MJPEG stream responsive)
# ---------------------------------------------------------------------------
class InferenceWorker(threading.Thread):
    """Pulls the newest frame from `get_frame()` and runs counter.process().

    get_frame: callable returning an HxWx3 uint8 numpy array (or None).
    CPU inference is slower than the old on-sensor path, so this always works
    on the LATEST frame and drops older ones instead of queueing them.
    """

    def __init__(self, counter, get_frame, min_interval=0.0):
        super().__init__(daemon=True)
        self.counter = counter
        self.get_frame = get_frame
        self.min_interval = min_interval
        self._stop_evt = threading.Event()
        self.last_ms = 0.0

    def run(self):
        while not self._stop_evt.is_set():
            t0 = time.time()
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                self.counter.process(frame)
            except Exception as exc:  # keep the worker alive
                print(f"[ncnn] inference error: {exc}")
                time.sleep(0.1)
            self.last_ms = (time.time() - t0) * 1000.0
            sleep_for = self.min_interval - (time.time() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def stop(self):
        self._stop_evt.set()


if __name__ == "__main__":
    # Quick standalone check: python3 ncnn_detector.py some_image.jpg
    import sys

    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else None
    if img is None:
        sys.exit("usage: python3 ncnn_detector.py <image>")
    det = NcnnDetector(frame_is_bgr=True)  # cv2.imread is BGR
    t0 = time.time()
    results = det.detect(img)
    print(f"{len(results)} detections in {(time.time() - t0) * 1000:.0f} ms")
    for r in results:
        print(f"  box=({r[0]:.0f},{r[1]:.0f},{r[2]:.0f},{r[3]:.0f}) conf={r[4]:.2f} cls={r[5]}")
