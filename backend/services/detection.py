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
_RUNTIME_CACHE_DIR = os.path.join(_BACKEND_DIR, ".cache")


def is_git_lfs_pointer(path: str) -> bool:
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


def build_yolo_model_candidates():
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
for _candidate in build_yolo_model_candidates():
    if not _candidate:
        continue
    if os.path.isabs(_candidate):
        if os.path.isfile(_candidate) and not is_git_lfs_pointer(_candidate):
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
KNOWN_FACE_ANCHOR_WINDOW_SEC = 1.5
KNOWN_FACE_ANCHOR_SEARCH_EXPAND = 1.85
KNOWN_FACE_STALE_ANCHOR_MAX_GAP_SEC = 10.0
KNOWN_FACE_STALE_ANCHOR_SEARCH_EXPAND = 3.25

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


def get_face_net():
    """Load the res10_300x300_ssd face detector (~10.7 MB Caffe model)."""
    global _face_net
    if _face_net is None:
        _face_net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)
        logger.info("Loaded res10_300x300_ssd_iter_140000 for face detection (~10.7 MB)")
    return _face_net


def ensure_runtime_cache_dirs():
    """Give model-side helpers a writable cache directory inside the project."""
    cache_dirs = {
        "MPLCONFIGDIR": os.path.join(_RUNTIME_CACHE_DIR, "matplotlib"),
        "XDG_CACHE_HOME": os.path.join(_RUNTIME_CACHE_DIR, "xdg-cache"),
    }
    for env_name, path in cache_dirs.items():
        current = (os.environ.get(env_name) or "").strip()
        if current:
            os.makedirs(current, exist_ok=True)
            continue
        os.makedirs(path, exist_ok=True)
        os.environ[env_name] = path


def build_insightface_provider_candidates():
    configured = (os.environ.get("INSIGHTFACE_PROVIDERS") or "").strip()
    if configured:
        parsed = [item.strip() for item in configured.split(",") if item.strip()]
        candidates = [parsed] if parsed else []
    elif sys.platform == "darwin":
        # CoreML has been unstable here and can leak runtime resources on shutdown.
        # Default to CPU on macOS and let CoreML be opt-in via INSIGHTFACE_PROVIDERS.
        candidates = [["CPUExecutionProvider"]]
    else:
        candidates = [["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]

    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception:
        available = None

    filtered_candidates = []
    seen = set()
    for provider_group in candidates:
        filtered = tuple(
            provider
            for provider in provider_group
            if available is None or provider in available
        )
        if not filtered or filtered in seen:
            continue
        seen.add(filtered)
        filtered_candidates.append(list(filtered))

    if not filtered_candidates:
        filtered_candidates.append(["CPUExecutionProvider"])
    return filtered_candidates


def get_face_app():
    """Load InsightFace buffalo_l for primary face detection plus ArcFace embeddings."""
    global _face_app, _face_app_load_failed
    if _face_app_load_failed:
        return None
    if _face_app is None:
        ensure_runtime_cache_dirs()
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            _face_app = None
            _face_app_load_failed = True
            logger.warning(
                "InsightFace unavailable; falling back to res10-only face detection. "
                "Manual blur tracking will still run, but face embeddings/supplemental detections are disabled: %s",
                e,
            )
            return None

        last_error = None
        provider_candidates = build_insightface_provider_candidates()
        for providers in provider_candidates:
            try:
                app = FaceAnalysis(
                    name="buffalo_l",
                    providers=providers,
                )
                app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
                _face_app = app
                logger.info(
                    "Loaded InsightFace buffalo_l for face detection and embeddings (providers=%s)",
                    ",".join(providers),
                )
                return _face_app
            except Exception as e:
                last_error = e
                logger.warning(
                    "InsightFace provider set %s failed; trying next option: %s",
                    ",".join(providers),
                    e,
                )

        _face_app = None
        _face_app_load_failed = True
        logger.warning(
            "InsightFace unavailable; falling back to res10-only face detection. "
            "Manual blur tracking will still run, but face embeddings/supplemental detections are disabled: %s",
            last_error,
        )
    return _face_app


def get_obj_model():
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


def face_sharpness(img_bgr, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def detect_faces_res10(img_bgr, confidence_threshold=RES10_CONFIDENCE):
    """Run res10_300x300_ssd face detector. Returns list of (x1,y1,x2,y2,conf)."""
    net = get_face_net()
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


def iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def get_embeddings_for_boxes(img_bgr, res10_boxes):
    """Run InsightFace on the frame, then match its detections to res10 boxes by IoU
    to assign 512-d ArcFace embeddings to each res10 detection.
    Also returns unmatched InsightFace detections (faces res10 missed)."""
    insight_faces = get_insightface_detections(img_bgr, with_encodings=True)
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
            overlap = iou(r_box[:4], ib)
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


def get_insightface_detections(img_bgr, with_encodings=False):
    app = get_face_app()
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
    insight_detections = get_insightface_detections(img_bgr, with_encodings=False)
    if insight_detections:
        results = []
        for det in insight_detections:
            x1, y1, x2, y2 = det["bbox"]
            sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
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

    res10_boxes = detect_faces_res10(img_bgr, confidence_threshold=confidence_threshold)

    results = []
    for (x1, y1, x2, y2, conf) in res10_boxes:
        sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
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

    _, unmatched_insight = get_embeddings_for_boxes(img_bgr, res10_boxes)
    for extra in unmatched_insight:
        x1, y1, x2, y2 = extra["bbox"]
        sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
        if sharpness < MIN_FACE_SHARPNESS:
            continue
        results.append({
            "bbox": [x1, y1, x2, y2],
            "det_score": extra["det_score"],
            "sharpness": round(sharpness, 1),
            "source": "insightface",
        })

    return results


def normalize_face_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def face_bbox_center_distance(box_a, box_b):
    if not box_a or not box_b:
        return 1e9
    acx = (float(box_a[0]) + float(box_a[2])) / 2.0
    acy = (float(box_a[1]) + float(box_a[3])) / 2.0
    bcx = (float(box_b[0]) + float(box_b[2])) / 2.0
    bcy = (float(box_b[1]) + float(box_b[3])) / 2.0
    return float(np.hypot(acx - bcx, acy - bcy))


def find_known_face_anchor_bbox(known_face, time_sec, max_gap_sec=KNOWN_FACE_ANCHOR_WINDOW_SEC):
    if time_sec is None:
        return None, None

    best_bbox = None
    best_gap = None
    for appearance in known_face.get("appearances") or []:
        bbox = normalize_face_bbox(appearance.get("bbox"))
        timestamp = appearance.get("timestamp")
        if bbox is None or timestamp is None:
            continue
        try:
            gap = abs(float(timestamp) - float(time_sec))
        except (TypeError, ValueError):
            continue
        if gap > max_gap_sec:
            continue
        if best_gap is None or gap < best_gap:
            best_bbox = bbox
            best_gap = gap

    return best_bbox, best_gap


def find_nearest_known_face_anchor_bbox(known_face, time_sec):
    if time_sec is None:
        return normalize_face_bbox(known_face.get("bbox")), None

    best_bbox = normalize_face_bbox(known_face.get("bbox"))
    best_gap = None
    for appearance in known_face.get("appearances") or []:
        bbox = normalize_face_bbox(appearance.get("bbox"))
        timestamp = appearance.get("timestamp")
        if bbox is None or timestamp is None:
            continue
        try:
            gap = abs(float(timestamp) - float(time_sec))
        except (TypeError, ValueError):
            continue
        if best_gap is None or gap < best_gap:
            best_bbox = bbox
            best_gap = gap

    return best_bbox, best_gap


def known_face_search_expand_factor(anchor_gap):
    if anchor_gap is None:
        return KNOWN_FACE_ANCHOR_SEARCH_EXPAND
    if anchor_gap <= KNOWN_FACE_ANCHOR_WINDOW_SEC:
        return KNOWN_FACE_ANCHOR_SEARCH_EXPAND
    return min(
        KNOWN_FACE_STALE_ANCHOR_SEARCH_EXPAND,
        KNOWN_FACE_ANCHOR_SEARCH_EXPAND + (anchor_gap - KNOWN_FACE_ANCHOR_WINDOW_SEC) * 0.24,
    )


def localize_known_faces_in_frame(frame_bgr, known_faces, time_sec=None, tolerance=0.35):
    """Locate specific saved faces in a frame.

    Prefers nearby saved appearance boxes as anchors so selected-face blur still
    works even when embedding-based identity matching is unavailable. Falls back
    to encoding matches when they are available.
    """
    if not known_faces:
        return []

    matched_by_person = {}
    if any(face.get("encoding") is not None for face in known_faces):
        for det in identify_faces_in_frame(frame_bgr, known_faces, tolerance=tolerance):
            person_id = str(det.get("person_id") or "").strip()
            if person_id:
                matched_by_person[person_id] = det

    frame_h, frame_w = frame_bgr.shape[:2]
    results = []
    used_boxes = []
    from services.redactor import detect_best_face_bbox, expand_bbox

    for known_face in known_faces:
        person_id = str(known_face.get("person_id") or "").strip()
        if not person_id:
            continue

        anchor_bbox, anchor_gap = find_known_face_anchor_bbox(known_face, time_sec)
        if anchor_bbox is None:
            anchor_bbox, anchor_gap = find_nearest_known_face_anchor_bbox(known_face, time_sec)
        anchor_candidate = None
        if anchor_bbox is not None:
            search_expand = known_face_search_expand_factor(anchor_gap)
            refined_bbox = detect_best_face_bbox(
                frame_bgr,
                expand_bbox(anchor_bbox, frame_w, frame_h, search_expand),
                preferred_bbox=anchor_bbox,
                allow_supplemental=True,
            )
            candidate_bbox = normalize_face_bbox(refined_bbox or anchor_bbox)
            if candidate_bbox is not None and (
                anchor_gap is None or anchor_gap <= KNOWN_FACE_STALE_ANCHOR_MAX_GAP_SEC
            ):
                confidence_penalty = 0.0 if anchor_gap is None else min(0.55, max(0.0, anchor_gap - 0.25) * 0.08)
                anchor_candidate = {
                    "bbox": candidate_bbox,
                    "person_id": person_id,
                    "match_score": round(max(0.18, 0.98 - confidence_penalty), 4),
                    "det_score": 0.58 if refined_bbox is not None else 0.34,
                }

        matched_candidate = matched_by_person.get(person_id)
        chosen = None
        if anchor_candidate and matched_candidate:
            overlap = iou(anchor_candidate["bbox"], matched_candidate["bbox"])
            distance = face_bbox_center_distance(anchor_candidate["bbox"], matched_candidate["bbox"])
            anchor_diag = max(
                1.0,
                float(np.hypot(
                    anchor_candidate["bbox"][2] - anchor_candidate["bbox"][0],
                    anchor_candidate["bbox"][3] - anchor_candidate["bbox"][1],
                )),
            )
            chosen = matched_candidate if (overlap >= 0.08 or distance <= anchor_diag * 0.7) else anchor_candidate
        else:
            chosen = anchor_candidate or matched_candidate

        if chosen is None:
            continue

        if any(iou(chosen["bbox"], existing) > 0.72 for existing in used_boxes):
            continue

        used_boxes.append(chosen["bbox"])
        results.append(chosen)

    return results


def detect_faces(img_bgr, with_encodings=False):
    """Detect faces using InsightFace first, with res10 fallback when unavailable.
    InsightFace provides stronger face localization and embeddings, which helps
    live blur alignment and person-specific matching remain stable.
    """
    insight_detections = get_insightface_detections(img_bgr, with_encodings=with_encodings)
    if insight_detections:
        results = []
        for det in insight_detections:
            x1, y1, x2, y2 = det["bbox"]
            sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
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

    res10_boxes = detect_faces_res10(img_bgr)

    embeddings = [None] * len(res10_boxes)
    unmatched_insight = []
    if with_encodings:
        if res10_boxes:
            embeddings, unmatched_insight = get_embeddings_for_boxes(img_bgr, res10_boxes)
        else:
            _, unmatched_insight = get_embeddings_for_boxes(img_bgr, [])

    results = []

    for i, (x1, y1, x2, y2, conf) in enumerate(res10_boxes):
        sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
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
        sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
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
        model = get_obj_model()
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
    Uses the same InsightFace-first detection pipeline as the rest of the
    product so person-specific blur stays aligned with generic face blur.
    """
    if not target_encodings:
        return []

    target_vecs = []
    for t in target_encodings:
        v = np.array(t, dtype=np.float32)
        norm = np.linalg.norm(v)
        target_vecs.append(v / norm if norm > 0 else v)

    matched = []
    for det in detect_faces(frame_bgr, with_encodings=True):
        enc = det.get("encoding")
        if enc is None:
            continue
        enc_arr = np.array(enc, dtype=np.float32)
        for t_enc in target_vecs:
            sim = float(np.dot(enc_arr, t_enc))
            if sim >= (1.0 - tolerance):
                x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [])]
                matched.append((x1, y1, x2, y2))
                break

    return matched


def match_objects_in_frame(frame_bgr, target_classes, conf_threshold=0.25, strict=True):
    if not target_classes:
        return []
    try:
        model = get_obj_model()
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
    best_by_person = {}
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
            candidate = {
                "bbox": det.get("bbox"),
                "person_id": best_person_id,
                "match_score": round(best_sim, 4),
                "det_score": round(float(det.get("det_score", 0.0)), 4),
            }
            previous = best_by_person.get(best_person_id)
            if previous is None or (
                candidate["match_score"],
                candidate["det_score"],
            ) > (
                previous["match_score"],
                previous["det_score"],
            ):
                best_by_person[best_person_id] = candidate

    return list(best_by_person.values())
