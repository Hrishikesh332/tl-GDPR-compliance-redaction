"""Resolve model assets and validate the production image during its build."""
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO

backend = Path(__file__).resolve().parents[1]
object_weights = backend / "yolov8n.pt"
if object_weights.exists() and object_weights.stat().st_size < 1024:
    if object_weights.read_bytes().startswith(b"version https://git-lfs.github.com/"):
        object_weights.unlink()

for weights in (object_weights, backend / "models/yolov8n-face-lindevs.pt"):
    model = YOLO(str(weights))
    model.predict(np.zeros((64, 64, 3), dtype=np.uint8), device="cpu", verbose=False)

cv2.dnn.readNetFromCaffe(
    str(backend / "models/deploy.prototxt"),
    str(backend / "models/res10_300x300_ssd_iter_140000.caffemodel"),
)
assert hasattr(cv2, "legacy"), "OpenCV contrib tracking support is required"

# Bundle the app's existing InsightFace model to avoid downloads on first use.
face_model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_model.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
print("Production face/object model smoke checks passed")
