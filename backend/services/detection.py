import logging
import os
import sys
import warnings

import cv2
import numpy as np

from utils.image import crop_to_base64, crop_face_to_base64, crop_with_bbox_to_base64

warnings.filterwarnings("ignore", category=FutureWarning, message=".*rcond.*")

logger = logging.getLogger("video_redaction.detection")

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROTOTXT = os.path.join(_BACKEND_DIR, "models", "deploy.prototxt")
_CAFFEMODEL = os.path.join(_BACKEND_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")

_face_net = None
_face_app = None
_obj_model = None

MIN_FACE_SIZE = 30
MIN_FACE_SHARPNESS = 10.0
RES10_CONFIDENCE = 0.35

_ONNX_PROVIDERS = (
    ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if sys.platform == "darwin"
    else ["CUDAExecutionProvider", "CPUExecutionProvider"]
)

FORENSIC_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle",
    "knife", "scissors", "gun",
    "cell phone", "laptop",
    "handbag", "backpack", "suitcase",
}


def _get_face_net():
    """Load the res10_300x300_ssd face detector (~10.7 MB Caffe model)."""
    global _face_net
    if _face_net is None:
        _face_net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)
        logger.info("Loaded res10_300x300_ssd_iter_140000 for face detection (~10.7 MB)")
    return _face_net


def _get_face_app():
    """Load InsightFace buffalo_l — used only for ArcFace 512-d embeddings, not detection."""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name="buffalo_l",
            providers=_ONNX_PROVIDERS,
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        logger.info("Loaded InsightFace buffalo_l (ArcFace embeddings only)")
    return _face_app


def _get_obj_model():
    global _obj_model
    if _obj_model is None:
        from ultralytics import YOLO
        _obj_model = YOLO("yolov8x.pt")
        logger.info("Loaded YOLOv8x for object detection")
    return _obj_model


def _face_sharpness(img_bgr, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _detect_faces_res10(img_bgr, confidence_threshold=RES10_CONFIDENCE):
    """Run res10_300x300_ssd face detector. Returns list of (x1,y1,x2,y2,conf)."""
    net = _get_face_net()
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < confidence_threshold:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 >= MIN_FACE_SIZE and y2 - y1 >= MIN_FACE_SIZE:
            boxes.append((x1, y1, x2, y2, conf))

    return boxes


def _iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _get_embeddings_for_boxes(img_bgr, res10_boxes):
    """Run InsightFace on the frame, then match its detections to res10 boxes by IoU
    to assign 512-d ArcFace embeddings to each res10 detection.
    Also returns unmatched InsightFace detections (faces res10 missed)."""
    app = _get_face_app()
    insight_faces = app.get(img_bgr)

    embeddings = [None] * len(res10_boxes)
    matched_insight_indices = set()

    for idx, r_box in enumerate(res10_boxes):
        best_iou = 0.3
        best_emb = None
        best_j = -1
        for j, iface in enumerate(insight_faces):
            ib = iface.bbox.astype(int).tolist()
            overlap = _iou(r_box[:4], ib)
            if overlap > best_iou and iface.embedding is not None:
                best_iou = overlap
                norm = np.linalg.norm(iface.embedding)
                best_emb = (iface.embedding / norm).tolist() if norm > 0 else iface.embedding.tolist()
                best_j = j
        embeddings[idx] = best_emb
        if best_j >= 0:
            matched_insight_indices.add(best_j)

    unmatched = []
    for j, iface in enumerate(insight_faces):
        if j in matched_insight_indices:
            continue
        if iface.embedding is None:
            continue
        ib = iface.bbox.astype(int).tolist()
        x1, y1, x2, y2 = ib
        if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
            continue
        det_score = float(iface.det_score) if hasattr(iface, "det_score") else 0.5
        norm = np.linalg.norm(iface.embedding)
        emb = (iface.embedding / norm).tolist() if norm > 0 else iface.embedding.tolist()
        unmatched.append({
            "bbox": [x1, y1, x2, y2],
            "det_score": round(det_score, 4),
            "embedding": emb,
        })

    return embeddings, unmatched


def detect_faces(img_bgr, with_encodings=False):
    """Detect faces using res10 + InsightFace (supplemental).
    InsightFace's RetinaFace catches faces that res10 misses (side profiles,
    smaller faces, partial occlusions). Filters out tiny and blurry faces.
    """
    res10_boxes = _detect_faces_res10(img_bgr)

    embeddings = [None] * len(res10_boxes)
    unmatched_insight = []
    if with_encodings:
        if res10_boxes:
            embeddings, unmatched_insight = _get_embeddings_for_boxes(img_bgr, res10_boxes)
        else:
            _, unmatched_insight = _get_embeddings_for_boxes(img_bgr, [])

    results = []

    for i, (x1, y1, x2, y2, conf) in enumerate(res10_boxes):
        sharpness = _face_sharpness(img_bgr, (x1, y1, x2, y2))
        if sharpness < MIN_FACE_SHARPNESS:
            continue

        fw, fh = x2 - x1, y2 - y1
        crop_area = fw * fh
        snap_b64 = crop_face_to_base64(img_bgr, (x1, y1, x2, y2))

        entry = {
            "bbox": [x1, y1, x2, y2],
            "snap_base64": snap_b64,
            "identification": "face",
            "crop_area": crop_area,
            "det_score": round(conf, 4),
            "sharpness": round(sharpness, 1),
        }
        if with_encodings:
            entry["encoding"] = embeddings[i]
        results.append(entry)

    for extra in unmatched_insight:
        x1, y1, x2, y2 = extra["bbox"]
        sharpness = _face_sharpness(img_bgr, (x1, y1, x2, y2))
        if sharpness < MIN_FACE_SHARPNESS:
            continue

        fw, fh = x2 - x1, y2 - y1
        crop_area = fw * fh
        snap_b64 = crop_face_to_base64(img_bgr, (x1, y1, x2, y2))

        entry = {
            "bbox": [x1, y1, x2, y2],
            "snap_base64": snap_b64,
            "identification": "face",
            "crop_area": crop_area,
            "det_score": extra["det_score"],
            "sharpness": round(sharpness, 1),
        }
        if with_encodings:
            entry["encoding"] = extra["embedding"]
        results.append(entry)

    if unmatched_insight:
        logger.info("InsightFace found %d additional faces missed by res10", len(unmatched_insight))

    return results


def detect_objects(img_bgr, conf_threshold=0.25, forensic_only=True):
    model = _get_obj_model()
    results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
    out = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            if forensic_only and name not in FORENSIC_CLASSES:
                continue
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            snap_b64 = crop_with_bbox_to_base64(
                img_bgr, xyxy, label=name, confidence=conf,
            )
            x1, y1, x2, y2 = xyxy
            crop_area = abs(x2 - x1) * abs(y2 - y1)
            out.append({
                "bbox": [round(x, 2) for x in xyxy],
                "snap_base64": snap_b64,
                "identification": name,
                "confidence": round(conf, 4),
                "crop_area": crop_area,
            })
    return out


def match_faces_in_frame(frame_bgr, target_encodings, tolerance=0.35):
    """Match faces in a frame against target encodings.
    Uses res10 + InsightFace supplemental for detection, ArcFace for comparison.
    """
    if not target_encodings:
        return []

    res10_boxes = _detect_faces_res10(frame_bgr, confidence_threshold=0.4)

    embeddings, unmatched_insight = _get_embeddings_for_boxes(frame_bgr, res10_boxes)

    all_boxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in res10_boxes]
    all_embeddings = list(embeddings)

    for extra in unmatched_insight:
        all_boxes.append(tuple(extra["bbox"]))
        all_embeddings.append(extra["embedding"])

    target_vecs = []
    for t in target_encodings:
        v = np.array(t, dtype=np.float32)
        norm = np.linalg.norm(v)
        target_vecs.append(v / norm if norm > 0 else v)

    matched = []
    for i, (x1, y1, x2, y2) in enumerate(all_boxes):
        enc = all_embeddings[i]
        if enc is None:
            continue
        enc_arr = np.array(enc, dtype=np.float32)
        for t_enc in target_vecs:
            sim = float(np.dot(enc_arr, t_enc))
            if sim >= (1.0 - tolerance):
                matched.append((x1, y1, x2, y2))
                break

    return matched


def match_objects_in_frame(frame_bgr, target_classes, conf_threshold=0.25):
    if not target_classes:
        return []
    model = _get_obj_model()
    results = model.predict(frame_bgr, conf=conf_threshold, verbose=False, imgsz=640)
    matched = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            if name in target_classes:
                matched.append(tuple(box.xyxy[0].tolist()))
    return matched
