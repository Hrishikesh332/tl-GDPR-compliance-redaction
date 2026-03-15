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


def _is_git_lfs_pointer(path: str) -> bool:
    """Return True if the file looks like a Git LFS pointer stub (not real weights)."""
    try:
        if not os.path.isfile(path):
            return False
        # LFS pointer files are tiny and start with: "version https://git-lfs.github.com/spec/v1"
        if os.path.getsize(path) < 1024:  # real YOLO weights are many MB
            with open(path, "rb") as f:
                head = f.read(64)
            return head.startswith(b"version https://git-lfs.github.com/")
        return False
    except Exception:
        return False


def _build_yolo_model_candidates():
    configured = (os.environ.get("YOLO_OBJECT_MODEL") or "").strip()
    if configured:
        return [
            configured if os.path.isabs(configured) else os.path.join(_BACKEND_DIR, configured),
            configured,
        ]

    # Default to the checked-in local weights for responsiveness and offline use.
    # Newer Ultralytics weights can still be opted into via YOLO_OBJECT_MODEL.
    return [
        os.path.join(_BACKEND_DIR, "yolov8n.pt"),
        "yolov8n.pt",
        os.path.join(_BACKEND_DIR, "yolo11n.pt"),
        "yolo11n.pt",
    ]


_YOLO_MODEL_CANDIDATES = []
for _candidate in _build_yolo_model_candidates():
    if not _candidate:
        continue
    if os.path.isabs(_candidate):
        if os.path.isfile(_candidate) and not _is_git_lfs_pointer(_candidate):
            _YOLO_MODEL_CANDIDATES.append(_candidate)
    else:
        _YOLO_MODEL_CANDIDATES.append(_candidate)

if not _YOLO_MODEL_CANDIDATES:
    _YOLO_MODEL_CANDIDATES = ["yolov8n.pt"]

_face_net = None
_face_app = None
_face_app_load_failed = False
_obj_model = None
_obj_model_load_failed = False
_obj_model_error = None

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


class ObjectDetectionUnavailable(RuntimeError):
    """Raised when the YOLO object detector cannot be loaded in this environment."""


def get_object_detection_error():
    return _obj_model_error


def _get_face_net():
    """Load the res10_300x300_ssd face detector (~10.7 MB Caffe model)."""
    global _face_net
    if _face_net is None:
        _face_net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)
        logger.info("Loaded res10_300x300_ssd_iter_140000 for face detection (~10.7 MB)")
    return _face_net


def _get_face_app():
    """Load InsightFace buffalo_l for primary face detection plus ArcFace embeddings."""
    global _face_app, _face_app_load_failed
    if _face_app_load_failed:
        return None
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
            _face_app = FaceAnalysis(
                name="buffalo_l",
                providers=_ONNX_PROVIDERS,
            )
            _face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
            logger.info("Loaded InsightFace buffalo_l for face detection and embeddings")
        except Exception as e:
            _face_app = None
            _face_app_load_failed = True
            logger.warning(
                "InsightFace unavailable; falling back to res10-only face detection. "
                "Manual blur tracking will still run, but face embeddings/supplemental detections are disabled: %s",
                e,
            )
    return _face_app


def _get_obj_model():
    global _obj_model, _obj_model_load_failed, _obj_model_error
    if _obj_model_load_failed:
        raise ObjectDetectionUnavailable(_obj_model_error or "YOLO object detection is unavailable.")
    if _obj_model is None:
        tried = []
        last_error = None
        try:
            from ultralytics import YOLO
        except Exception as e:
            _obj_model = None
            _obj_model_load_failed = True
            _obj_model_error = (
                "YOLO object detection could not start because Ultralytics failed to import in the active Python "
                "environment. Make sure the backend is running from backend/.venv and restart the server."
            )
            logger.warning("%s Underlying error: %s", _obj_model_error, e)
            raise ObjectDetectionUnavailable(_obj_model_error) from e

        for candidate in _YOLO_MODEL_CANDIDATES:
            display_name = os.path.basename(candidate) if os.path.isabs(candidate) else candidate
            try:
                _obj_model = YOLO(candidate)
                logger.info("Loaded YOLO model for object detection: %s", display_name)
                _obj_model_error = None
                return _obj_model
            except Exception as e:
                last_error = e
                tried.append(display_name)
                logger.warning("Failed loading YOLO model candidate %s: %s", display_name, e)

        _obj_model = None
        _obj_model_load_failed = True
        tried_display = ", ".join(tried) if tried else "none"
        _obj_model_error = (
            "YOLO object detection could not start. Tried these model candidates: "
            f"{tried_display}. Ensure the backend is running from backend/.venv and that newer Ultralytics "
            "weights can be downloaded when needed, then restart the backend."
        )
        logger.warning("%s Underlying error: %s", _obj_model_error, last_error)
        raise ObjectDetectionUnavailable(_obj_model_error) from last_error
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
    insight_faces = _get_insightface_detections(img_bgr, with_encodings=True)
    if not insight_faces:
        return [None] * len(res10_boxes), []

    embeddings = [None] * len(res10_boxes)
    matched_insight_indices = set()

    for idx, r_box in enumerate(res10_boxes):
        best_iou = 0.3
        best_emb = None
        best_j = -1
        for j, iface in enumerate(insight_faces):
            ib = iface["bbox"]
            overlap = _iou(r_box[:4], ib)
            if overlap > best_iou and iface.get("embedding") is not None:
                best_iou = overlap
                best_emb = iface["embedding"]
                best_j = j
        embeddings[idx] = best_emb
        if best_j >= 0:
            matched_insight_indices.add(best_j)

    unmatched = []
    for j, iface in enumerate(insight_faces):
        if j in matched_insight_indices:
            continue
        if iface.get("embedding") is None:
            continue
        x1, y1, x2, y2 = iface["bbox"]
        if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
            continue
        unmatched.append({
            "bbox": [x1, y1, x2, y2],
            "det_score": iface["det_score"],
            "embedding": iface["embedding"],
        })

    return embeddings, unmatched


def _get_insightface_detections(img_bgr, with_encodings=False):
    app = _get_face_app()
    if app is None:
        return []

    detections = []
    for iface in app.get(img_bgr):
        bbox = iface.bbox.astype(int).tolist()
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
            continue

        det_score = float(iface.det_score) if hasattr(iface, "det_score") else 0.5
        entry = {
            "bbox": [x1, y1, x2, y2],
            "det_score": round(det_score, 4),
        }
        if with_encodings and getattr(iface, "embedding", None) is not None:
            norm = np.linalg.norm(iface.embedding)
            entry["embedding"] = (iface.embedding / norm).tolist() if norm > 0 else iface.embedding.tolist()
        detections.append(entry)

    return detections


def detect_face_boxes(img_bgr, confidence_threshold=RES10_CONFIDENCE, include_supplemental=False):
    """Return face boxes with confidence metadata, without encoding snapshots."""
    insight_detections = _get_insightface_detections(img_bgr, with_encodings=False)
    if insight_detections:
        results = []
        for det in insight_detections:
            x1, y1, x2, y2 = det["bbox"]
            sharpness = _face_sharpness(img_bgr, (x1, y1, x2, y2))
            if sharpness < MIN_FACE_SHARPNESS:
                continue
            results.append({
                "bbox": [x1, y1, x2, y2],
                "det_score": det["det_score"],
                "sharpness": round(sharpness, 1),
                "source": "insightface",
            })
        if results:
            return results

    res10_boxes = _detect_faces_res10(img_bgr, confidence_threshold=confidence_threshold)

    results = []
    for (x1, y1, x2, y2, conf) in res10_boxes:
        sharpness = _face_sharpness(img_bgr, (x1, y1, x2, y2))
        if sharpness < MIN_FACE_SHARPNESS:
            continue
        results.append({
            "bbox": [x1, y1, x2, y2],
            "det_score": round(conf, 4),
            "sharpness": round(sharpness, 1),
            "source": "res10",
        })

    if not include_supplemental:
        return results

    _, unmatched_insight = _get_embeddings_for_boxes(img_bgr, res10_boxes)
    for extra in unmatched_insight:
        x1, y1, x2, y2 = extra["bbox"]
        sharpness = _face_sharpness(img_bgr, (x1, y1, x2, y2))
        if sharpness < MIN_FACE_SHARPNESS:
            continue
        results.append({
            "bbox": [x1, y1, x2, y2],
            "det_score": extra["det_score"],
            "sharpness": round(sharpness, 1),
            "source": "insightface",
        })

    return results


def detect_faces(img_bgr, with_encodings=False):
    """Detect faces using InsightFace first, with res10 fallback when unavailable.
    InsightFace provides stronger face localization and embeddings, which helps
    live blur alignment and person-specific matching remain stable.
    """
    insight_detections = _get_insightface_detections(img_bgr, with_encodings=with_encodings)
    if insight_detections:
        results = []
        for det in insight_detections:
            x1, y1, x2, y2 = det["bbox"]
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
                "det_score": det["det_score"],
                "sharpness": round(sharpness, 1),
            }
            if with_encodings:
                entry["encoding"] = det.get("embedding")
            results.append(entry)

        if results:
            return results

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


def detect_objects(img_bgr, conf_threshold=0.25, forensic_only=True, strict=True):
    try:
        model = _get_obj_model()
    except ObjectDetectionUnavailable:
        if strict:
            raise
        return []
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


def match_objects_in_frame(frame_bgr, target_classes, conf_threshold=0.25, strict=True):
    if not target_classes:
        return []
    try:
        model = _get_obj_model()
    except ObjectDetectionUnavailable:
        if strict:
            raise
        return []
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


def identify_faces_in_frame(frame_bgr, known_faces, tolerance=0.35):
    """Identify faces in a frame against known face encodings.

    known_faces: iterable of {"person_id": str, "encoding": [...]}
    Returns [{"bbox": [...], "person_id": str, "match_score": float, "det_score": float}, ...]
    """
    if not known_faces:
        return []

    prepared_known = []
    for face in known_faces:
        person_id = str(face.get("person_id") or "").strip()
        encoding = face.get("encoding")
        if not person_id or encoding is None:
            continue
        vec = np.array(encoding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        prepared_known.append((person_id, vec / norm if norm > 0 else vec))

    if not prepared_known:
        return []

    detections = detect_faces(frame_bgr, with_encodings=True)
    identified = []
    for det in detections:
        enc = det.get("encoding")
        if enc is None:
            continue
        enc_arr = np.array(enc, dtype=np.float32)
        best_person_id = None
        best_sim = -1.0
        for person_id, known_vec in prepared_known:
            sim = float(np.dot(enc_arr, known_vec))
            if sim > best_sim:
                best_sim = sim
                best_person_id = person_id
        if best_person_id and best_sim >= (1.0 - tolerance):
            identified.append({
                "bbox": det.get("bbox"),
                "person_id": best_person_id,
                "match_score": round(best_sim, 4),
                "det_score": round(float(det.get("det_score", 0.0)), 4),
            })

    return identified
