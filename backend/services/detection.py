import logging
import math
import os
import sys
import warnings

import cv2
import numpy as np

from services.face_identity import get_face_identity
from utils.image import crop_to_base64, crop_face_to_base64, crop_with_bbox_to_base64

warnings.filterwarnings("ignore", category=FutureWarning, message=".*rcond.*")

logger = logging.getLogger("video_redaction.detection")

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROTOTXT = os.path.join(_BACKEND_DIR, "models", "deploy.prototxt")
_CAFFEMODEL = os.path.join(_BACKEND_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
_RUNTIME_CACHE_DIR = os.path.join(_BACKEND_DIR, ".cache")


def is_git_lfs_pointer(path: str) -> bool:
    try:
        if not os.path.isfile(path):
            return False
        if os.path.getsize(path) < 1024:  
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
SMALL_FACE_MIN_SIZE = 10
MIN_FACE_SHARPNESS = 10.0
SMALL_FACE_SHARPNESS = 3.0
RES10_CONFIDENCE = 0.35
# Snap-face frame picker (/api/detect-faces): maximize recall; user picks one crop.
SNAP_UPLOAD_PICKER_CONFIDENCE = 0.22
SNAP_UPLOAD_PICKER_MIN_FACE = 22
SNAP_UPLOAD_PICKER_MIN_SHARPNESS = 3.25
SNAP_UPLOAD_PICKER_UPSCALE = 1.42
FAR_FACE_UPSCALE = 2.2
KNOWN_FACE_ANCHOR_WINDOW_SEC = 1.5
KNOWN_FACE_ANCHOR_SEARCH_EXPAND = 1.85
KNOWN_FACE_STALE_ANCHOR_MAX_GAP_SEC = 10.0
KNOWN_FACE_STALE_ANCHOR_SEARCH_EXPAND = 3.25
PERSON_HEAD_CONFIDENCE = 0.18
PERSON_HEAD_MIN_SIDE = 8

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
    global _face_net
    if _face_net is None:
        _face_net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)
        logger.info("Loaded res10_300x300_ssd_iter_140000 for face detection (~10.7 MB)")
    return _face_net


def ensure_runtime_cache_dirs():
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


def detect_faces_res10(
    img_bgr,
    confidence_threshold=RES10_CONFIDENCE,
    min_face_size=MIN_FACE_SIZE,
    upscale=1.0,
):
    """Run res10_300x300_ssd face detector. Returns list of (x1,y1,x2,y2,conf)."""
    net = get_face_net()
    scale = max(1.0, float(upscale or 1.0))
    detector_img = img_bgr
    if scale > 1.01:
        src_h, src_w = img_bgr.shape[:2]
        detector_img = cv2.resize(
            img_bgr,
            (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )

    h, w = detector_img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        detector_img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
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
        if scale > 1.01:
            x1 = int(round(x1 / scale))
            y1 = int(round(y1 / scale))
            x2 = int(round(x2 / scale))
            y2 = int(round(y2 / scale))
            out_h, out_w = img_bgr.shape[:2]
        else:
            out_h, out_w = h, w
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out_w, x2), min(out_h, y2)
        if x2 - x1 >= min_face_size and y2 - y1 >= min_face_size:
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


def get_embeddings_for_boxes(
    img_bgr,
    res10_boxes,
    min_face_size=MIN_FACE_SIZE,
    upscale=1.0,
):

    # Run InsightFace on the frame, then match its detections to res10 boxes by IoU
    # to assign 512-d ArcFace embeddings to each res10 detection.
    # Also returns unmatched InsightFace detections (faces res10 missed).
    insight_faces = get_insightface_detections(
        img_bgr,
        with_encodings=True,
        min_face_size=min_face_size,
        upscale=upscale,
    )
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
        if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
            continue
        unmatched.append({
            "bbox": [x1, y1, x2, y2],
            "det_score": iface["det_score"],
            "embedding": iface["embedding"],
        })

    return embeddings, unmatched


def get_insightface_detections(
    img_bgr,
    with_encodings=False,
    min_face_size=MIN_FACE_SIZE,
    upscale=1.0,
):
    app = get_face_app()
    if app is None:
        return []

    scale = max(1.0, float(upscale or 1.0))
    detector_img = img_bgr
    if scale > 1.01:
        src_h, src_w = img_bgr.shape[:2]
        detector_img = cv2.resize(
            img_bgr,
            (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )

    detections = []
    out_h, out_w = img_bgr.shape[:2]
    for iface in app.get(detector_img):
        bbox = iface.bbox.astype(int).tolist()
        x1, y1, x2, y2 = bbox
        if scale > 1.01:
            x1 = int(round(x1 / scale))
            y1 = int(round(y1 / scale))
            x2 = int(round(x2 / scale))
            y2 = int(round(y2 / scale))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out_w, x2), min(out_h, y2)
        if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
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


def detect_face_boxes(
    img_bgr,
    confidence_threshold=RES10_CONFIDENCE,
    include_supplemental=False,
    min_face_size=MIN_FACE_SIZE,
    min_sharpness=MIN_FACE_SHARPNESS,
    upscale=1.0,
):

    insight_detections = get_insightface_detections(
        img_bgr,
        with_encodings=False,
        min_face_size=min_face_size,
        upscale=upscale,
    )
    if insight_detections:
        results = []
        for det in insight_detections:
            x1, y1, x2, y2 = det["bbox"]
            sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
            if sharpness < min_sharpness:
                continue
            results.append({
                "bbox": [x1, y1, x2, y2],
                "det_score": det["det_score"],
                "sharpness": round(sharpness, 1),
                "source": "insightface",
            })
        if results:
            return results

    res10_boxes = detect_faces_res10(
        img_bgr,
        confidence_threshold=confidence_threshold,
        min_face_size=min_face_size,
        upscale=upscale,
    )

    results = []
    for (x1, y1, x2, y2, conf) in res10_boxes:
        sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
        if sharpness < min_sharpness:
            continue
        results.append({
            "bbox": [x1, y1, x2, y2],
            "det_score": round(conf, 4),
            "sharpness": round(sharpness, 1),
            "source": "res10",
        })

    if not include_supplemental:
        return results

    _, unmatched_insight = get_embeddings_for_boxes(
        img_bgr,
        res10_boxes,
        min_face_size=min_face_size,
        upscale=upscale,
    )
    for extra in unmatched_insight:
        x1, y1, x2, y2 = extra["bbox"]
        sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
        if sharpness < min_sharpness:
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


def face_bbox_area(bbox):
    if not bbox:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def prepare_face_encoding_vector(encoding):
    if encoding is None:
        return None
    try:
        vec = np.array(encoding, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vec.size == 0:
        return None
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def face_bbox_center_distance(box_a, box_b):
    if not box_a or not box_b:
        return 1e9
    acx = (float(box_a[0]) + float(box_a[2])) / 2.0
    acy = (float(box_a[1]) + float(box_a[3])) / 2.0
    bcx = (float(box_b[0]) + float(box_b[2])) / 2.0
    bcy = (float(box_b[1]) + float(box_b[3])) / 2.0
    return float(np.hypot(acx - bcx, acy - bcy))


def normalize_frame_bbox(bbox, frame_w, frame_h):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    x1 = max(0.0, min(x1, float(frame_w)))
    y1 = max(0.0, min(y1, float(frame_h)))
    x2 = max(0.0, min(x2, float(frame_w)))
    y2 = max(0.0, min(y2, float(frame_h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def infer_head_bbox_from_person_bbox(person_bbox, frame_w, frame_h):
    """Infer a head region from a YOLO person box.

    This is used only as a fallback when the selected face turns away or
    becomes too small for a frontal-face detector. It intentionally returns the
    upper part of the person box, not the whole person, so the blur still feels
    like a face/head redaction.
    """
    person_bbox = normalize_frame_bbox(person_bbox, frame_w, frame_h)
    if person_bbox is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in person_bbox]
    person_w = max(1.0, x2 - x1)
    person_h = max(1.0, y2 - y1)
    if person_w < PERSON_HEAD_MIN_SIDE or person_h < PERSON_HEAD_MIN_SIDE:
        return None

    # Far-away people need a little more vertical coverage because the
    # detector's top edge often lands on hair/hood pixels rather than forehead,
    # but keep this tight: the fallback is for a head/face blur, not a torso.
    head_h_ratio = 0.30 if person_h < 96 else 0.26 if person_h < 180 else 0.22
    head_h = max(float(PERSON_HEAD_MIN_SIDE), person_h * head_h_ratio)
    head_w = max(float(PERSON_HEAD_MIN_SIDE), min(person_w * 0.62, max(person_w * 0.42, head_h * 0.82)))
    cx = (x1 + x2) / 2.0
    hy1 = y1 - person_h * 0.01
    hy2 = y1 + head_h + person_h * 0.02
    return normalize_frame_bbox(
        (cx - head_w / 2.0, hy1, cx + head_w / 2.0, hy2),
        frame_w,
        frame_h,
    )


def fit_head_bbox_to_preferred(head_bbox, preferred_bbox, frame_w, frame_h):
    """Keep a person-derived head fallback close to the tracked face size."""
    head_bbox = normalize_frame_bbox(head_bbox, frame_w, frame_h)
    preferred_bbox = normalize_face_bbox(preferred_bbox)
    if head_bbox is None or preferred_bbox is None:
        return head_bbox

    hx1, hy1, hx2, hy2 = [float(v) for v in head_bbox]
    px1, py1, px2, py2 = [float(v) for v in preferred_bbox]
    hw = max(1.0, hx2 - hx1)
    hh = max(1.0, hy2 - hy1)
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    hcx = (hx1 + hx2) / 2.0
    hcy = (hy1 + hy2) / 2.0
    pref_diag = max(1.0, float(np.hypot(pw, ph)))
    center_shift = float(np.hypot(hcx - pcx, hcy - pcy)) / pref_diag

    # If the person-box head is close to the tracker, mostly trust the
    # tracker center; otherwise allow a stronger recenter but keep the size
    # bounded by the tracked face/head dimensions.
    preferred_bias = 0.72 if center_shift <= 0.8 else 0.56
    cx = pcx * preferred_bias + hcx * (1.0 - preferred_bias)
    cy = pcy * preferred_bias + hcy * (1.0 - preferred_bias)
    target_w = min(max(pw * 1.10, float(PERSON_HEAD_MIN_SIDE)), max(pw * 1.42, float(PERSON_HEAD_MIN_SIDE)), hw)
    target_h = min(max(ph * 1.16, float(PERSON_HEAD_MIN_SIDE)), max(ph * 1.52, float(PERSON_HEAD_MIN_SIDE)), hh)
    return normalize_frame_bbox(
        (cx - target_w / 2.0, cy - target_h / 2.0, cx + target_w / 2.0, cy + target_h / 2.0),
        frame_w,
        frame_h,
    )


def localize_head_in_search_region(
    frame_bgr,
    search_bbox,
    preferred_bbox=None,
    conf_threshold=PERSON_HEAD_CONFIDENCE,
    strict=True,
):
    """Find a likely head near ``preferred_bbox`` using YOLO person boxes.

    Face detectors cannot see the back of a head. This helper keeps the
    redaction attached to the selected face's expected location by searching
    for nearby person boxes and converting the best person's upper region into
    a head bbox. It returns ``None`` when geometry is ambiguous.
    """
    if frame_bgr is None or search_bbox is None:
        return None

    frame_h, frame_w = frame_bgr.shape[:2]
    search = normalize_frame_bbox(search_bbox, frame_w, frame_h)
    if search is None:
        return None
    preferred = normalize_frame_bbox(preferred_bbox, frame_w, frame_h) or search
    preferred_area = max(1.0, face_bbox_area(preferred))
    preferred_diag = max(
        1.0,
        float(np.hypot(preferred[2] - preferred[0], preferred[3] - preferred[1])),
    )
    search_diag = max(
        1.0,
        float(np.hypot(search[2] - search[0], search[3] - search[1])),
    )

    try:
        model = get_obj_model()
    except ObjectDetectionUnavailable:
        return None

    try:
        results = model.predict(frame_bgr, conf=conf_threshold, verbose=False, imgsz=768)
    except Exception:
        return None

    candidates = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] != "person":
                continue
            person_bbox = normalize_frame_bbox(box.xyxy[0].tolist(), frame_w, frame_h)
            head_bbox = infer_head_bbox_from_person_bbox(person_bbox, frame_w, frame_h)
            if preferred_bbox is not None:
                head_bbox = fit_head_bbox_to_preferred(head_bbox, preferred, frame_w, frame_h)
            if head_bbox is None:
                continue

            head_search_iou = iou(head_bbox, search)
            person_search_iou = iou(person_bbox, search) if person_bbox is not None else 0.0
            search_center_ratio = face_bbox_center_distance(head_bbox, search) / search_diag
            preferred_iou = iou(head_bbox, preferred)
            preferred_center_ratio = face_bbox_center_distance(head_bbox, preferred) / preferred_diag
            area_penalty = abs(math.log(max(1.0, face_bbox_area(head_bbox)) / preferred_area))

            max_search_center = 0.78 if strict else 1.05
            max_preferred_center = 0.72 if strict else 1.0
            if head_search_iou < 0.02 and person_search_iou < 0.02 and search_center_ratio > max_search_center:
                continue
            if preferred_iou < 0.03 and preferred_center_ratio > max_preferred_center:
                continue
            if strict and area_penalty > 1.25 and preferred_iou < 0.08:
                continue

            det_score = float(box.conf[0]) if box.conf is not None else conf_threshold
            geometry_score = (
                det_score * 1.4
                + preferred_iou * 4.2
                + head_search_iou * 2.0
                + person_search_iou * 0.6
                - preferred_center_ratio * 1.55
                - search_center_ratio * 0.55
                - area_penalty * 0.28
            )
            candidates.append({
                "bbox": head_bbox,
                "person_bbox": person_bbox,
                "det_score": round(det_score, 4),
                "geometry_score": geometry_score,
                "source": "person_head",
            })

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["geometry_score"], reverse=True)
    best = candidates[0]
    if strict and len(candidates) > 1 and (best["geometry_score"] - candidates[1]["geometry_score"]) < 0.45:
        return None
    return best


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


def localize_known_face_in_search_region(
    frame_bgr,
    known_face,
    search_bbox,
    preferred_bbox=None,
    tolerance=0.55,
    allow_geometry_fallback=False,
):
    """
    This keeps person-specific blur efficient by only searching a cropped area,
    while preferring the selected person's embedding when available. When the
    embedding model/runtime drifts across environments, a tight anchor can still
    fall back to strict geometry checks inside the selected search region.
    """
    if frame_bgr is None or known_face is None:
        return None

    normalized_search_bbox = normalize_face_bbox(search_bbox)
    if normalized_search_bbox is None:
        return None

    person_id = get_face_identity(known_face)
    if not person_id:
        return None

    frame_h, frame_w = frame_bgr.shape[:2]
    sx1, sy1, sx2, sy2 = normalized_search_bbox
    sx1 = max(0, min(sx1, frame_w))
    sy1 = max(0, min(sy1, frame_h))
    sx2 = max(0, min(sx2, frame_w))
    sy2 = max(0, min(sy2, frame_h))
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    crop = frame_bgr[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return None

    preferred_bbox = normalize_face_bbox(preferred_bbox) or normalize_face_bbox(known_face.get("bbox"))
    preferred_area = max(1.0, face_bbox_area(preferred_bbox)) if preferred_bbox else None
    preferred_diag = None
    if preferred_bbox:
        preferred_diag = max(
            1.0,
            float(np.hypot(
                preferred_bbox[2] - preferred_bbox[0],
                preferred_bbox[3] - preferred_bbox[1],
            )),
        )

    known_vec = prepare_face_encoding_vector(known_face.get("encoding"))
    detections = detect_faces(
        crop,
        with_encodings=known_vec is not None,
        confidence_threshold=0.16,
        min_face_size=SMALL_FACE_MIN_SIZE,
        min_sharpness=2.0,
        upscale=FAR_FACE_UPSCALE,
    )

    identity_best = None
    identity_best_score = -1e9
    geometry_candidates = []

    for det in detections:
        det_bbox = normalize_face_bbox([
            sx1 + float(det["bbox"][0]),
            sy1 + float(det["bbox"][1]),
            sx1 + float(det["bbox"][2]),
            sy1 + float(det["bbox"][3]),
        ])
        if det_bbox is None:
            continue

        det_score = float(det.get("det_score", 0.0))
        overlap = iou(det_bbox, preferred_bbox) if preferred_bbox else 0.0
        center_distance = face_bbox_center_distance(det_bbox, preferred_bbox) if preferred_bbox else 0.0
        center_ratio = (center_distance / preferred_diag) if preferred_diag else 0.0
        area_penalty = 0.0
        if preferred_area:
            det_area = max(1.0, face_bbox_area(det_bbox))
            area_penalty = abs(math.log(det_area / preferred_area))

        geometry_score = (
            det_score * 1.65
            + overlap * 4.1
            - center_ratio * 1.45
            - area_penalty * 0.45
        )
        geometry_candidate = {
            "bbox": det_bbox,
            "person_id": person_id,
            "match_score": round(max(0.12, min(0.74, 0.18 + overlap * 0.42)), 4),
            "det_score": round(det_score, 4),
            "geometry_score": geometry_score,
            "overlap": overlap,
            "center_ratio": center_ratio,
            "area_penalty": area_penalty,
        }
        geometry_candidates.append(geometry_candidate)

        if known_vec is None:
            continue

        det_vec = prepare_face_encoding_vector(det.get("encoding"))
        if det_vec is None:
            continue

        similarity = float(np.dot(det_vec, known_vec))
        if similarity < (1.0 - tolerance):
            continue

        identity_score = geometry_score + similarity * 5.0
        if identity_score > identity_best_score:
            identity_best_score = identity_score
            identity_best = {
                "bbox": det_bbox,
                "person_id": person_id,
                "match_score": round(similarity, 4),
                "det_score": round(det_score, 4),
            }

    if identity_best is not None:
        return identity_best

    if not allow_geometry_fallback or not geometry_candidates:
        return None

    geometry_candidates.sort(key=lambda candidate: candidate["geometry_score"], reverse=True)
    best_geometry = geometry_candidates[0]
    next_best_score = geometry_candidates[1]["geometry_score"] if len(geometry_candidates) > 1 else None
    isolated = next_best_score is None or (best_geometry["geometry_score"] - next_best_score) >= 0.85

    if known_vec is not None:
        if not isolated:
            return None
        if best_geometry["overlap"] < 0.48 and best_geometry["center_ratio"] > 0.16:
            return None
        if best_geometry["area_penalty"] > 0.9:
            return None
        return {
            "bbox": best_geometry["bbox"],
            "person_id": person_id,
            "match_score": best_geometry["match_score"],
            "det_score": best_geometry["det_score"],
        }

    if best_geometry["overlap"] >= 0.34 or best_geometry["center_ratio"] <= 0.24:
        return {
            "bbox": best_geometry["bbox"],
            "person_id": person_id,
            "match_score": best_geometry["match_score"],
            "det_score": best_geometry["det_score"],
        }

    return None


def localize_known_faces_in_frame(frame_bgr, known_faces, time_sec=None, tolerance=0.55):
    """Locate specific saved faces in a frame.

    Prefers nearby saved appearance boxes as anchors so selected-face blur still
    works even when embedding-based identity matching is unavailable or drifts
    between environments. Falls back to broader full-frame identity matching
    only when an anchored search window cannot re-lock the face.
    """
    if not known_faces:
        return []

    frame_h, frame_w = frame_bgr.shape[:2]
    anchor_candidate_by_person = {}
    faces_needing_identity_match = []
    results = []
    used_boxes = []
    from services.redactor import expand_bbox

    for known_face in known_faces:
        person_id = get_face_identity(known_face)
        if not person_id:
            continue

        anchor_bbox, anchor_gap = find_known_face_anchor_bbox(known_face, time_sec)
        if anchor_bbox is None:
            anchor_bbox, anchor_gap = find_nearest_known_face_anchor_bbox(known_face, time_sec)
        anchor_candidate = None
        if anchor_bbox is not None:
            search_expand = known_face_search_expand_factor(anchor_gap)
            # Keep the face-specific search anchored near the saved appearance.
            # Even when a stored embedding exists, allow the stricter geometry
            # fallback in this narrow window so deploy/runtime model drift does
            # not silently disable person-specific blur.
            anchor_candidate = localize_known_face_in_search_region(
                frame_bgr,
                known_face=known_face,
                search_bbox=expand_bbox(anchor_bbox, frame_w, frame_h, search_expand),
                preferred_bbox=anchor_bbox,
                tolerance=tolerance,
                allow_geometry_fallback=True,
            )
            if anchor_candidate is not None and (
                anchor_gap is None or anchor_gap <= KNOWN_FACE_STALE_ANCHOR_MAX_GAP_SEC
            ):
                confidence_penalty = 0.0 if anchor_gap is None else min(0.48, max(0.0, anchor_gap - 0.25) * 0.07)
                anchor_candidate["match_score"] = round(
                    max(float(anchor_candidate.get("match_score", 0.0)), 0.98 - confidence_penalty),
                    4,
                )
                anchor_candidate["det_score"] = round(
                    max(float(anchor_candidate.get("det_score", 0.0)), 0.34),
                    4,
                )

        if anchor_candidate is not None:
            anchor_candidate_by_person[person_id] = anchor_candidate
            continue

        if known_face.get("encoding") is not None:
            faces_needing_identity_match.append(known_face)

    matched_by_person = {}
    if faces_needing_identity_match:
        # A nearby saved appearance anchor is much faster than running full
        # frame-wide face identification. Only fall back to embedding matches
        # for faces that could not be re-locked from their stored appearances.
        for det in identify_faces_in_frame(frame_bgr, faces_needing_identity_match, tolerance=tolerance):
            person_id = str(det.get("person_id") or "").strip()
            if person_id:
                matched_by_person[person_id] = det

    for known_face in known_faces:
        person_id = get_face_identity(known_face)
        if not person_id:
            continue

        anchor_candidate = anchor_candidate_by_person.get(person_id)
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


def detect_faces(
    img_bgr,
    with_encodings=False,
    confidence_threshold=RES10_CONFIDENCE,
    min_face_size=MIN_FACE_SIZE,
    min_sharpness=MIN_FACE_SHARPNESS,
    upscale=1.0,
):
    """Detect faces using InsightFace first, with res10 fallback when unavailable.
    InsightFace provides stronger face localization and embeddings, which helps
    live blur alignment and person-specific matching remain stable.
    """
    insight_detections = get_insightface_detections(
        img_bgr,
        with_encodings=with_encodings,
        min_face_size=min_face_size,
        upscale=upscale,
    )
    if insight_detections:
        results = []
        for det in insight_detections:
            x1, y1, x2, y2 = det["bbox"]
            sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
            if sharpness < min_sharpness:
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

    res10_boxes = detect_faces_res10(
        img_bgr,
        confidence_threshold=confidence_threshold,
        min_face_size=min_face_size,
        upscale=upscale,
    )

    embeddings = [None] * len(res10_boxes)
    unmatched_insight = []
    if with_encodings:
        if res10_boxes:
            embeddings, unmatched_insight = get_embeddings_for_boxes(
                img_bgr,
                res10_boxes,
                min_face_size=min_face_size,
                upscale=upscale,
            )
        else:
            _, unmatched_insight = get_embeddings_for_boxes(
                img_bgr,
                [],
                min_face_size=min_face_size,
                upscale=upscale,
            )

    results = []

    for i, (x1, y1, x2, y2, conf) in enumerate(res10_boxes):
        sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
        if sharpness < min_sharpness:
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
        if sharpness < min_sharpness:
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


def detect_uploaded_reference_faces(
    img_bgr,
    with_encodings=False,
    confidence_threshold=RES10_CONFIDENCE,
    min_face_size=MIN_FACE_SIZE,
    min_sharpness=MIN_FACE_SHARPNESS,
    detect_upscale=1.0,
):

    try:
        res10_boxes = detect_faces_res10(
            img_bgr,
            confidence_threshold=confidence_threshold,
            min_face_size=min_face_size,
            upscale=detect_upscale,
        )
    except Exception as error:
        logger.warning("Uploaded-face ResNet-10 detection failed; falling back to general detector: %s", error)
        fallback_results = detect_faces(img_bgr, with_encodings=with_encodings)
        for result in fallback_results:
            result.setdefault("source", "fallback")
        return fallback_results

    # Always correlate with InsightFace so we merge ResNet misses (even when embeddings
    # are not needed for the caller).
    embeddings = [None] * len(res10_boxes)
    unmatched_insight = []
    if res10_boxes:
        embeddings, unmatched_insight = get_embeddings_for_boxes(
            img_bgr,
            res10_boxes,
            min_face_size=min_face_size,
            upscale=detect_upscale,
        )
    else:
        _, unmatched_insight = get_embeddings_for_boxes(
            img_bgr,
            [],
            min_face_size=min_face_size,
            upscale=detect_upscale,
        )
    if not with_encodings:
        embeddings = [None] * len(res10_boxes)

    results = []
    for index, (x1, y1, x2, y2, confidence) in enumerate(res10_boxes):
        sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
        if sharpness < min_sharpness:
            continue

        entry = {
            "bbox": [x1, y1, x2, y2],
            "det_score": round(float(confidence), 4),
            "sharpness": round(sharpness, 1),
            "source": "res10",
        }
        if with_encodings:
            entry["encoding"] = embeddings[index]
        results.append(entry)

    for extra in unmatched_insight:
        x1, y1, x2, y2 = extra["bbox"]
        sharpness = face_sharpness(img_bgr, (x1, y1, x2, y2))
        if sharpness < min_sharpness:
            continue

        entry = {
            "bbox": [x1, y1, x2, y2],
            "det_score": round(float(extra.get("det_score", 0.0)), 4),
            "sharpness": round(sharpness, 1),
            "source": "insightface_supplemental",
        }
        if with_encodings:
            entry["encoding"] = extra.get("embedding")
        results.append(entry)

    if results:
        return results

    fallback_results = detect_faces(img_bgr, with_encodings=with_encodings)
    for result in fallback_results:
        result.setdefault("source", "fallback")
    return fallback_results


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


def match_faces_in_frame(frame_bgr, target_encodings, tolerance=0.55):
    """Match faces in a frame against target encodings.
    Uses the same InsightFace-first detection pipeline as the rest of the
    product so person-specific blur stays aligned with generic face blur.
    tolerance: max allowed (1 - cosine_similarity).  Higher = more permissive.
    """
    if not target_encodings:
        return []

    target_vecs = []
    for t in target_encodings:
        v = np.array(t, dtype=np.float32)
        norm = np.linalg.norm(v)
        target_vecs.append(v / norm if norm > 0 else v)

    matched = []
    for det in detect_faces(
        frame_bgr,
        with_encodings=True,
        confidence_threshold=0.2,
        min_face_size=SMALL_FACE_MIN_SIZE,
        min_sharpness=SMALL_FACE_SHARPNESS,
        upscale=1.6,
    ):
        enc = det.get("encoding")
        if enc is None:
            continue
        enc_arr = np.array(enc, dtype=np.float32)
        norm = np.linalg.norm(enc_arr)
        if norm > 0:
            enc_arr = enc_arr / norm
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


def identify_faces_in_frame(frame_bgr, known_faces, tolerance=0.55):
    """Identify faces in a frame against known face encodings.

    known_faces: iterable of {"person_id": str, "encoding": [...]}
    Returns [{"bbox": [...], "person_id": str, "match_score": float, "det_score": float}, ...]
    tolerance: max allowed (1 - cosine_similarity).  Higher = more permissive.
    """
    if not known_faces:
        return []

    prepared_known = []
    for face in known_faces:
        person_id = get_face_identity(face)
        encoding = face.get("encoding")
        if not person_id or encoding is None:
            continue
        vec = prepare_face_encoding_vector(encoding)
        if vec is None:
            continue
        prepared_known.append((person_id, vec))

    if not prepared_known:
        return []

    detections = detect_faces(
        frame_bgr,
        with_encodings=True,
        confidence_threshold=0.2,
        min_face_size=SMALL_FACE_MIN_SIZE,
        min_sharpness=SMALL_FACE_SHARPNESS,
        upscale=1.6,
    )
    best_by_person = {}
    for det in detections:
        enc = det.get("encoding")
        if enc is None:
            continue
        enc_arr = prepare_face_encoding_vector(enc)
        if enc_arr is None:
            continue
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
