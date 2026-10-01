from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np

try:
    import ncnn
except ImportError as exc:  # pragma: no cover - imported only at runtime
    raise RuntimeError(
        "ncnn is required. Install it with: pip install ncnn"
    ) from exc


class NcnnShrimpCounter:
    """YOLO-style ncnn object detector + count-line tracker for shrimp.

    This mirrors the current app's logic: resize/letterbox to 640x640,
    run the ncnn model, decode the YOLO outputs, and count objects as they
    cross a horizontal count line. The detector expects a YOLOv8-style export
    produced by NCNN, with an input size of 640x640.
    """

    def __init__(
        self,
        model_param: str,
        model_bin: str,
        labels_path: str | None = None,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        num_threads: int = 4,
        input_name: str = "images",
        output_names: Tuple[str, ...] = ("output",),
    ):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.labels = self._load_labels(labels_path) if labels_path else ["shrimp"]
        self.input_name = input_name
        self.output_names = output_names

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = num_threads
        self.net.load_param(model_param)
        self.net.load_model(model_bin)
        self.extractor = self.net.create_extractor()

        self.tracked_objects: dict[int, dict] = {}
        self.next_object_id = 0
        self.total_count = 0

    def _load_labels(self, labels_path: str) -> List[str]:
        with open(labels_path, "r", encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]
        return labels or ["shrimp"]

    def preprocess(self, frame_bgr: np.ndarray) -> Tuple[ncnn.Mat, dict]:
        """Letterbox + normalize a frame to 640x640 for the ncnn model."""
        h, w = frame_bgr.shape[:2]
        target_w, target_h = self.input_size

        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0

        mat = ncnn.Mat.from_pixels(
            rgb,
            ncnn.Mat_PIXEL_RGB,
            target_w,
            target_h,
        )

        meta = {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "new_w": new_w,
            "new_h": new_h,
            "original_size": (w, h),
        }
        return mat, meta

    def _extract_output_tensors(self):
        out_list = []
        for name in self.output_names:
            try:
                _, mat = self.extractor.extract(name)
            except Exception:
                continue
            if mat is not None:
                out_list.append(mat)
        return out_list

    def _mat_to_numpy(self, mat: ncnn.Mat) -> np.ndarray:
        try:
            return np.array(mat)
        except Exception:
            pass

        if hasattr(mat, "to_numpy"):
            try:
                return mat.to_numpy()
            except Exception:
                pass

        if hasattr(mat, "clone"):
            try:
                clone = mat.clone()
                return np.array(clone)
            except Exception:
                pass

        raise TypeError(f"Could not convert ncnn.Mat to numpy for {type(mat)}")

    def _decode_yolov8(self, pred: np.ndarray, meta: dict):
        """Decode a YOLOv8-style NCNN output to boxes / scores."""
        if pred.size == 0:
            return []

        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 3:
            pred = pred.reshape(-1, pred.shape[-1])

        # YOLOv8 style output: [cx, cy, w, h, obj_conf, cls_0, cls_1, ...]
        box_columns = pred[:, :4]
        class_scores = pred[:, 4:]

        if class_scores.shape[1] == 0:
            return []

        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
        valid = max_scores >= self.conf_threshold

        detections = []
        for idx in np.where(valid)[0]:
            cx, cy, w, h = box_columns[idx]
            score = float(max_scores[idx])
            cls_id = int(class_ids[idx])

            if w <= 0 or h <= 0:
                continue

            x1 = (cx - w / 2.0)
            y1 = (cy - h / 2.0)
            x2 = (cx + w / 2.0)
            y2 = (cy + h / 2.0)

            x1 = (x1 - meta["offset_x"]) / meta["scale"]
            y1 = (y1 - meta["offset_y"]) / meta["scale"]
            x2 = (x2 - meta["offset_x"]) / meta["scale"]
            y2 = (y2 - meta["offset_y"]) / meta["scale"]

            out_w, out_h = meta["original_size"]
            x1 = max(0, min(out_w, x1))
            y1 = max(0, min(out_h, y1))
            x2 = max(0, min(out_w, x2))
            y2 = max(0, min(out_h, y2))

            detections.append(
                {
                    "box": (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    "conf": score,
                    "category": cls_id,
                }
            )

        return detections

    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        """Run a forward pass and return raw detection dicts."""
        mat, meta = self.preprocess(frame_bgr)

        try:
            self.extractor.input(self.input_name, mat)
        except Exception:
            # Some NCNN exports use a different input name; try the common
            # fallback names when needed.
            for name in ("data", "input", "images"):
                try:
                    self.extractor.input(name, mat)
                    self.input_name = name
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError("Could not bind the NCNN input tensor.")

        outputs = self._extract_output_tensors()
        if not outputs:
            return []

        candidates = []
        for out in outputs:
            arr = np.asarray(self._mat_to_numpy(out), dtype=np.float32)
            if arr.size == 0:
                continue

            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim == 3:
                # Common NCNN YOLO output layout: [1, N, C] -> flatten to [N, C]
                arr = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim > 2:
                arr = arr.reshape(-1, arr.shape[-1])

            if arr.shape[-1] < 4:
                continue
            candidates.append(arr)

        if not candidates:
            return []

        pred = np.concatenate(candidates, axis=0)
        if pred.shape[0] == 0:
            return []

        return self._decode_yolov8(pred, meta)

    def _count_crossing_objects(self, detections: List[dict], count_line_y_fraction: float = 0.62):
        """Track centroid movement and increment the count when an object crosses the line."""
        height = int(self.input_size[1] * count_line_y_fraction)
        split_y = height

        current_centroids = []
        for det in detections:
            x, y, w, h = det["box"]
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            current_centroids.append((cx, cy, x, y, w, h))

        if not current_centroids:
            for obj_id in list(self.tracked_objects.keys()):
                self.tracked_objects[obj_id]["disappeared"] += 1
                if self.tracked_objects[obj_id]["disappeared"] > 8:
                    del self.tracked_objects[obj_id]
            return 0

        if not self.tracked_objects:
            for cx, cy, x, y, w, h in current_centroids:
                self.tracked_objects[self.next_object_id] = {
                    "centroid": (cx, cy),
                    "counted": cy > split_y,
                    "disappeared": 0,
                }
                self.next_object_id += 1
            return sum(1 for obj in self.tracked_objects.values() if obj["counted"])

        used_centroids = set()
        used_ids = set()
        distances = []
        for i, (cx, cy, x, y, w, h) in enumerate(current_centroids):
            for obj_id, data in self.tracked_objects.items():
                prev_cx, prev_cy = data["centroid"]
                distance = math.hypot(cx - prev_cx, cy - prev_cy)
                if distance <= 40:
                    distances.append((distance, obj_id, i))

        distances.sort(key=lambda item: item[0])

        for distance, obj_id, i in distances:
            if obj_id in used_ids or i in used_centroids:
                continue
            used_ids.add(obj_id)
            used_centroids.add(i)

            cx, cy = current_centroids[i][0], current_centroids[i][1]
            prev_cy = self.tracked_objects[obj_id]["centroid"][1]
            self.tracked_objects[obj_id]["centroid"] = (cx, cy)
            self.tracked_objects[obj_id]["disappeared"] = 0

            if prev_cy <= split_y and cy > split_y and not self.tracked_objects[obj_id]["counted"]:
                self.tracked_objects[obj_id]["counted"] = True
                self.total_count += 1

        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in used_ids:
                self.tracked_objects[obj_id]["disappeared"] += 1
                if self.tracked_objects[obj_id]["disappeared"] > 8:
                    del self.tracked_objects[obj_id]

        for i, (cx, cy, x, y, w, h) in enumerate(current_centroids):
            if i not in used_centroids:
                self.tracked_objects[self.next_object_id] = {
                    "centroid": (cx, cy),
                    "counted": cy > split_y,
                    "disappeared": 0,
                }
                self.next_object_id += 1

        return self.total_count

    def count_frame(self, frame_bgr: np.ndarray, count_line_y_fraction: float = 0.62) -> Tuple[int, List[dict]]:
        """Convenience method that returns (counted_total, detections)."""
        detections = self.detect(frame_bgr)
        counted = self._count_crossing_objects(detections, count_line_y_fraction)
        return counted, detections


if __name__ == "__main__":
    detector = NcnnShrimpCounter(
        model_param="models/shrimp_yolov8.param",
        model_bin="models/shrimp_yolov8.bin",
        labels_path="models/labels.txt",
        conf_threshold=0.25,
        input_name="images",
        output_names=("output",),
    )

    frame = cv2.imread("test.jpg")
    if frame is None:
        raise FileNotFoundError("Place a sample image at test.jpg to test the NCNN detector.")

    count, detections = detector.count_frame(frame)
    print(f"Detected shrimp count: {count}")
    print(f"Detections: {len(detections)}")
