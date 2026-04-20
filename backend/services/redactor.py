import logging
import math
import os
import sys
import tempfile

import cv2
import numpy as np

from config import (
    TRACKER_MAX_DIM,
    MIN_TRACKER_ROI_PIXELS,
    DEFAULT_BLUR_STRENGTH,
    DEFAULT_DETECT_EVERY_N,
    TRACKER_REINIT_BBOX_EXPAND_FACTOR,
    TRACKER_SMOOTHING_ALPHA,
    MANUAL_FACE_SEARCH_EXPAND_FACTOR,
    MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR,
    MANUAL_FACE_DETECTION_CONFIDENCE,
)
from services.face_identity import get_face_identity
from utils.image import apply_redaction
from utils.video import small_frame_for_tracking, reencode_mp4_to_h264

logger = logging.getLogger("video_redaction.redactor")

FACE_REDACTION_PAD_X_RATIO = 0.14
FACE_REDACTION_PAD_TOP_RATIO = 0.24
FACE_REDACTION_PAD_BOTTOM_RATIO = 0.12
_NO_TRACKER_FACTORY = object()
_TRACKER_FACTORY_CACHE = {}
_TRACKER_FACTORY_LOGGED = set()
_TRACKER_UNAVAILABLE_WARNING_EMITTED = False


def tracker_factory_candidates(scale_adaptive=False):
    """Return tracker constructors ordered by preference for this OpenCV build."""
    legacy = getattr(cv2, "legacy", None)
    to_try = []
    if scale_adaptive:
        if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
            to_try.append(("CSRT.legacy", legacy.TrackerCSRT_create))
        if hasattr(cv2, "TrackerCSRT_create"):
            to_try.append(("CSRT", cv2.TrackerCSRT_create))
    if legacy is not None and hasattr(legacy, "TrackerKCF_create"):
        to_try.append(("KCF.legacy", legacy.TrackerKCF_create))
    if hasattr(cv2, "TrackerKCF_create"):
        to_try.append(("KCF", cv2.TrackerKCF_create))
    if legacy is not None and hasattr(legacy, "TrackerMOSSE_create"):
        to_try.append(("MOSSE.legacy", legacy.TrackerMOSSE_create))
    if hasattr(cv2, "TrackerMOSSE_create"):
        to_try.append(("MOSSE", cv2.TrackerMOSSE_create))
    # MIL is available in more OpenCV builds than CSRT/KCF/MOSSE and is still
    # better than repeatedly falling back to purely static tracking.
    if legacy is not None and hasattr(legacy, "TrackerMIL_create"):
        to_try.append(("MIL.legacy", legacy.TrackerMIL_create))
    if hasattr(cv2, "TrackerMIL_create"):
        to_try.append(("MIL", cv2.TrackerMIL_create))
    return to_try


def resolve_tracker_factory(scale_adaptive=False):
    key = bool(scale_adaptive)
    cached = _TRACKER_FACTORY_CACHE.get(key)
    if cached is not None:
        return None if cached is _NO_TRACKER_FACTORY else cached

    for name, create in tracker_factory_candidates(scale_adaptive=scale_adaptive):
        try:
            t = create()
            if t is not None:
                _TRACKER_FACTORY_CACHE[key] = (name, create)
                if name not in _TRACKER_FACTORY_LOGGED:
                    logger.info("Using tracker: %s for motion tracking", name)
                    _TRACKER_FACTORY_LOGGED.add(name)
                return name, create
        except (AttributeError, cv2.error, Exception):
            continue

    global _TRACKER_UNAVAILABLE_WARNING_EMITTED
    _TRACKER_FACTORY_CACHE[key] = _NO_TRACKER_FACTORY
    if not _TRACKER_UNAVAILABLE_WARNING_EMITTED:
        logger.warning(
            "No preferred OpenCV tracker was available. Tracking will fall back to optical-flow/static tracking."
        )
        _TRACKER_UNAVAILABLE_WARNING_EMITTED = True
    return None


def create_tracker(scale_adaptive=False):
    """Create a tracker instance using the best constructor available in this OpenCV build."""
    resolved = resolve_tracker_factory(scale_adaptive=scale_adaptive)
    if resolved is None:
        return None

    _name, create = resolved
    try:
        return create()
    except (AttributeError, cv2.error, Exception):
        # If a cached constructor later becomes unusable, clear it so future calls
        # can probe lower-priority trackers instead of retrying the same one forever.
        _TRACKER_FACTORY_CACHE.pop(bool(scale_adaptive), None)
    return None


def smooth_bbox(new_bbox, prev_bbox, alpha, frame_w, frame_h):
    """Exponential moving average of bbox. alpha = weight of new (0 = no smoothing, use raw)."""
    if alpha is None or alpha <= 0 or alpha >= 1 or prev_bbox is None:
        return new_bbox
    x1 = int(alpha * new_bbox[0] + (1 - alpha) * prev_bbox[0])
    y1 = int(alpha * new_bbox[1] + (1 - alpha) * prev_bbox[1])
    x2 = int(alpha * new_bbox[2] + (1 - alpha) * prev_bbox[2])
    y2 = int(alpha * new_bbox[3] + (1 - alpha) * prev_bbox[3])
    x1 = max(0, min(x1, frame_w))
    y1 = max(0, min(y1, frame_h))
    x2 = max(0, min(x2, frame_w))
    y2 = max(0, min(y2, frame_h))
    if x2 <= x1 or y2 <= y1:
        return new_bbox
    return (x1, y1, x2, y2)


def expand_bbox(bbox, frame_w, frame_h, factor):
    """Expand bbox by factor (center-out) and clamp to frame. factor > 1 e.g. 1.15."""
    if factor is None or factor <= 1.0:
        return bbox
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    hw = (x2 - x1) / 2
    hh = (y2 - y1) / 2
    hw *= factor
    hh *= factor
    x1 = max(0, int(cx - hw))
    y1 = max(0, int(cy - hh))
    x2 = min(frame_w, int(cx + hw))
    y2 = min(frame_h, int(cy + hh))
    if x2 <= x1 or y2 <= y1:
        return bbox
    return (x1, y1, x2, y2)


def expand_face_redaction_bbox(
    bbox,
    frame_w,
    frame_h,
    pad_x_ratio=FACE_REDACTION_PAD_X_RATIO,
    pad_top_ratio=FACE_REDACTION_PAD_TOP_RATIO,
    pad_bottom_ratio=FACE_REDACTION_PAD_BOTTOM_RATIO,
):
    if not bbox:
        return bbox

    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    expanded = (
        max(0, int(round(x1 - box_w * pad_x_ratio))),
        max(0, int(round(y1 - box_h * pad_top_ratio))),
        min(frame_w, int(round(x2 + box_w * pad_x_ratio))),
        min(frame_h, int(round(y2 + box_h * pad_bottom_ratio))),
    )
    if expanded[2] <= expanded[0] or expanded[3] <= expanded[1]:
        return bbox
    return expanded


def bbox_area(bbox):
    if not bbox:
        return 0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_iou(box_a, box_b):
    if not box_a or not box_b:
        return 0.0
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def bbox_center_distance(box_a, box_b):
    if not box_a or not box_b:
        return 1e9
    acx = (box_a[0] + box_a[2]) / 2.0
    acy = (box_a[1] + box_a[3]) / 2.0
    bcx = (box_b[0] + box_b[2]) / 2.0
    bcy = (box_b[1] + box_b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def is_face_track_motion_consistent(candidate_bbox, reference_bbox):
    if not candidate_bbox or not reference_bbox:
        return False
    if bbox_iou(candidate_bbox, reference_bbox) >= 0.42:
        return True
    ref_diag = max(
        1.0,
        math.hypot(
            reference_bbox[2] - reference_bbox[0],
            reference_bbox[3] - reference_bbox[1],
        ),
    )
    return bbox_center_distance(candidate_bbox, reference_bbox) <= ref_diag * 0.26


def frame_bbox_to_tracker_roi(bbox, scale_back, small_w, small_h):
    if not bbox or small_w <= 0 or small_h <= 0:
        return None
    x1, y1, x2, y2 = bbox
    sx = int(x1 / scale_back)
    sy = int(y1 / scale_back)
    sw = max(MIN_TRACKER_ROI_PIXELS, int((x2 - x1) / scale_back))
    sh = max(MIN_TRACKER_ROI_PIXELS, int((y2 - y1) / scale_back))
    sw = min(sw, small_w)
    sh = min(sh, small_h)
    if sw < 1 or sh < 1:
        return None
    sx = max(0, min(sx, small_w - sw))
    sy = max(0, min(sy, small_h - sh))
    return (sx, sy, sw, sh)


def init_tracker_from_frame_bbox(tracker, small_frame, bbox, scale_back):
    if tracker is None or bbox is None:
        return False
    small_h, small_w = small_frame.shape[:2]
    roi = frame_bbox_to_tracker_roi(bbox, scale_back, small_w, small_h)
    if roi is None:
        return False
    tracker.init(small_frame, roi)
    return True


def create_initialized_tracker(small_frame, bbox, scale_back, scale_adaptive=True):
    tracker = create_tracker(scale_adaptive=scale_adaptive)
    if tracker is None:
        return None
    try:
        if init_tracker_from_frame_bbox(tracker, small_frame, bbox, scale_back):
            return tracker
    except (cv2.error, AttributeError, Exception):
        return None
    return None


def custom_region_tracking_mode(region):
    mode = str(region.get("tracking_mode", "")).strip().lower()
    if mode in {"face", "generic"}:
        return mode
    reason = str(region.get("reason", "")).lower()
    if any(token in reason for token in ("face", "person", "head")):
        return "face"
    return "generic"


def face_padding_from_bbox(manual_bbox, face_bbox):
    if not manual_bbox or not face_bbox:
        return None
    fx1, fy1, fx2, fy2 = face_bbox
    fw = max(1, fx2 - fx1)
    fh = max(1, fy2 - fy1)
    mx1, my1, mx2, my2 = manual_bbox
    return {
        "left": (fx1 - mx1) / fw,
        "top": (fy1 - my1) / fh,
        "right": (mx2 - fx2) / fw,
        "bottom": (my2 - fy2) / fh,
    }


def apply_face_padding(face_bbox, padding, frame_w, frame_h):
    if not face_bbox:
        return None
    if not padding:
        return face_bbox
    x1, y1, x2, y2 = face_bbox
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)
    out = (
        int(round(x1 - padding.get("left", 0.0) * fw)),
        int(round(y1 - padding.get("top", 0.0) * fh)),
        int(round(x2 + padding.get("right", 0.0) * fw)),
        int(round(y2 + padding.get("bottom", 0.0) * fh)),
    )
    return tracker_roi_to_frame_bbox(out[0], out[1], out[2] - out[0], out[3] - out[1], 1.0, frame_w, frame_h)


def detect_best_face_bbox(frame, search_bbox, preferred_bbox=None, allow_supplemental=True):
    if not search_bbox:
        return None

    x1, y1, x2, y2 = [int(v) for v in search_bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    from services.detection import detect_face_boxes

    detections = detect_face_boxes(
        crop,
        confidence_threshold=MANUAL_FACE_DETECTION_CONFIDENCE,
        include_supplemental=allow_supplemental,
    )
    if not detections:
        return None

    best_bbox = None
    best_score = -1e9
    preferred_area = max(1, bbox_area(preferred_bbox)) if preferred_bbox else None
    preferred_diag = None
    if preferred_bbox:
        preferred_diag = max(
            1.0,
            math.hypot(preferred_bbox[2] - preferred_bbox[0], preferred_bbox[3] - preferred_bbox[1]),
        )

    for det in detections:
        dx1, dy1, dx2, dy2 = [int(v) for v in det["bbox"]]
        det_bbox = (x1 + dx1, y1 + dy1, x1 + dx2, y1 + dy2)
        score = float(det.get("det_score", 0.0)) * 2.0 + (bbox_area(det_bbox) / max(1.0, bbox_area(search_bbox))) * 0.2
        if preferred_bbox:
            iou = bbox_iou(det_bbox, preferred_bbox)
            center_penalty = bbox_center_distance(det_bbox, preferred_bbox) / preferred_diag
            area_penalty = abs(math.log(max(1.0, bbox_area(det_bbox)) / preferred_area))
            score += iou * 4.0
            score -= center_penalty * 1.35
            score -= area_penalty * 0.35
        if score > best_score:
            best_score = score
            best_bbox = det_bbox

    return best_bbox


def frame_bbox_to_small_bbox(bbox, scale_back, small_w, small_h):
    roi = frame_bbox_to_tracker_roi(bbox, scale_back, small_w, small_h)
    if roi is None:
        return None
    sx, sy, sw, sh = roi
    return (sx, sy, sx + sw, sy + sh)


def small_bbox_to_frame_bbox(bbox, scale_back, frame_w, frame_h):
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return tracker_roi_to_frame_bbox(x1, y1, x2 - x1, y2 - y1, scale_back, frame_w, frame_h)


def bbox_corners(bbox):
    x1, y1, x2, y2 = bbox
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)


def corners_to_bbox(corners, frame_w, frame_h, min_side=4):
    if corners is None or len(corners) == 0:
        return None
    xs = corners[:, 0, 0]
    ys = corners[:, 0, 1]
    x1 = max(0, min(int(np.floor(xs.min())), frame_w - 1))
    y1 = max(0, min(int(np.floor(ys.min())), frame_h - 1))
    x2 = max(0, min(int(np.ceil(xs.max())), frame_w))
    y2 = max(0, min(int(np.ceil(ys.max())), frame_h))
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    return (x1, y1, x2, y2)


def seed_tracking_points(gray_frame, bbox, max_corners=60):
    if gray_frame is None or bbox is None:
        return None
    h, w = gray_frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    points = cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
        mask=mask,
    )
    return points


def translate_bbox(bbox, dx, dy, frame_w, frame_h):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    moved = (
        int(round(x1 + dx)),
        int(round(y1 + dy)),
        int(round(x2 + dx)),
        int(round(y2 + dy)),
    )
    return corners_to_bbox(bbox_corners(moved), frame_w, frame_h)


def optical_flow_bbox_update(prev_gray, gray, prev_points, prev_bbox, frame_w, frame_h):
    if prev_gray is None or gray is None or prev_points is None or prev_bbox is None:
        return None, None
    if len(prev_points) < 2:
        return None, None

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if next_points is None or status is None:
        return None, None

    good_old = prev_points[status.flatten() == 1]
    good_new = next_points[status.flatten() == 1]
    if len(good_old) < 2 or len(good_new) < 2:
        return None, None

    transformed_points = good_new.reshape(-1, 1, 2)
    if len(good_old) >= 4:
        matrix, inliers = cv2.estimateAffinePartial2D(
            good_old.reshape(-1, 1, 2),
            good_new.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if matrix is not None:
            corners = cv2.transform(bbox_corners(prev_bbox), matrix)
            bbox = corners_to_bbox(corners, frame_w, frame_h)
            if bbox is not None:
                if inliers is not None:
                    mask = inliers.flatten() == 1
                    if mask.any():
                        transformed_points = good_new[mask].reshape(-1, 1, 2)
                return bbox, transformed_points

    diff = (good_new - good_old).reshape(-1, 2)
    delta = np.median(diff, axis=0)
    dx, dy = float(delta[0]), float(delta[1])
    bbox = translate_bbox(prev_bbox, dx, dy, frame_w, frame_h)
    if bbox is None:
        return None, None
    return bbox, transformed_points


def tracker_roi_to_frame_bbox(x, y, tw, th, scale_back, frame_w, frame_h, min_side=4):
    """Convert tracker ROI (x, y, w, h) in scaled space to clamped (x1, y1, x2, y2) in frame space.
    Ensures the blur area resizes with the tracker and stays within frame bounds."""
    fx = int(x * scale_back)
    fy = int(y * scale_back)
    fw = max(min_side, int(tw * scale_back))
    fh = max(min_side, int(th * scale_back))
    x1 = max(0, fx)
    y1 = max(0, fy)
    x2 = min(frame_w, fx + fw)
    y2 = min(frame_h, fy + fh)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def normalize_object_class_name(value):
    return str(value or "").strip().lower()


def match_objects(frame_bgr, target_classes, conf_threshold=0.25):
    if not target_classes:
        return []
    normalized_targets = {
        normalize_object_class_name(name)
        for name in target_classes
        if normalize_object_class_name(name)
    }
    if not normalized_targets:
        return []
    from services.detection import get_obj_model
    model = get_obj_model()
    results = model.predict(frame_bgr, conf=conf_threshold, verbose=False, imgsz=640)
    matched = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            if normalize_object_class_name(name) in normalized_targets:
                matched.append(tuple(box.xyxy[0].tolist()))
    return matched


def normalized_region_to_bbox(region, w, h):
    """Convert normalized (0-1) region to pixel bbox (x1, y1, x2, y2)."""
    x = float(region.get("x", 0))
    y = float(region.get("y", 0))
    width = float(region.get("width", 0.1))
    height = float(region.get("height", 0.1))
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + width) * w)
    y2 = int((y + height) * h)
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_to_normalized_region(bbox, frame_w, frame_h):
    if not bbox or frame_w <= 0 or frame_h <= 0:
        return None
    x1, y1, x2, y2 = bbox
    return {
        "x": max(0.0, min(1.0, x1 / frame_w)),
        "y": max(0.0, min(1.0, y1 / frame_h)),
        "width": max(0.0, min(1.0, (x2 - x1) / frame_w)),
        "height": max(0.0, min(1.0, (y2 - y1) / frame_h)),
    }


def initialize_auto_redaction_track(small_frame, gray_small, bbox, scale_back, kind, metadata=None):
    tracker = None
    try:
        tracker = create_initialized_tracker(small_frame, bbox, scale_back, scale_adaptive=True)
    except (cv2.error, AttributeError, Exception):
        tracker = None

    small_bbox = frame_bbox_to_small_bbox(
        bbox,
        scale_back,
        small_frame.shape[1],
        small_frame.shape[0],
    )
    return {
        **(metadata or {}),
        "kind": kind,
        "tracker": tracker,
        "last_bbox": bbox,
        "smoothed_bbox": bbox,
        "small_bbox": small_bbox,
        "points": seed_tracking_points(gray_small, small_bbox) if small_bbox is not None else None,
        "fail_count": 0,
    }


def update_auto_redaction_track(
    track,
    frame,
    small_frame,
    gray_small,
    prev_gray_small,
    scale_back,
    frame_w,
    frame_h,
    periodic_reinit=False,
    reinit_after_fails=5,
):
    tracker = track.get("tracker")
    kind = str(track.get("kind") or "object").strip().lower()
    known_face = track.get("known_face")
    requires_identity_lock = kind == "face" and known_face is not None
    identity_tolerance = track.get("identity_tolerance")
    last_bbox = track.get("last_bbox")
    prev_smoothed = track.get("smoothed_bbox") or last_bbox
    prev_small_bbox = track.get("small_bbox")
    prev_points = track.get("points")
    fail_count = int(track.get("fail_count", 0) or 0)
    small_h, small_w = small_frame.shape[:2]

    tracker_bbox = None
    tracker_ok = False
    optical_bbox = None
    optical_points = None
    optical_ok = False

    if tracker is not None and periodic_reinit and last_bbox:
        try:
            refreshed_tracker = create_initialized_tracker(
                small_frame,
                expand_bbox(last_bbox, frame_w, frame_h, TRACKER_REINIT_BBOX_EXPAND_FACTOR),
                scale_back,
                scale_adaptive=True,
            )
            if refreshed_tracker is not None:
                tracker = refreshed_tracker
                fail_count = 0
        except (cv2.error, AttributeError, Exception):
            pass

    if tracker is not None:
        ok, roi = tracker.update(small_frame)
        if ok and roi is not None:
            try:
                x, y, tw, th = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
                if tw > 0 and th > 0:
                    tracker_bbox = tracker_roi_to_frame_bbox(x, y, tw, th, scale_back, frame_w, frame_h)
                    tracker_ok = tracker_bbox is not None
            except (IndexError, TypeError, ValueError):
                tracker_ok = False

    if prev_gray_small is not None and prev_small_bbox is not None and prev_points is not None:
        optical_small_bbox, optical_points = optical_flow_bbox_update(
            prev_gray_small,
            gray_small,
            prev_points,
            prev_small_bbox,
            small_w,
            small_h,
        )
        if optical_small_bbox is not None:
            optical_bbox = small_bbox_to_frame_bbox(optical_small_bbox, scale_back, frame_w, frame_h)
            optical_ok = optical_bbox is not None

    if tracker is not None and optical_ok and (not tracker_ok or bbox_iou(tracker_bbox, optical_bbox) < 0.3):
        try:
            refreshed_tracker = create_initialized_tracker(
                small_frame,
                expand_bbox(optical_bbox, frame_w, frame_h, TRACKER_REINIT_BBOX_EXPAND_FACTOR),
                scale_back,
                scale_adaptive=True,
            )
            if refreshed_tracker is not None:
                tracker = refreshed_tracker
                fail_count = 0
        except (cv2.error, AttributeError, Exception):
            pass

    predicted_bbox = optical_bbox if optical_ok else (tracker_bbox if tracker_ok else None)
    resolved_bbox = predicted_bbox or last_bbox
    face_detected = False
    tracking_success = tracker_ok or optical_ok

    if kind == "face" and resolved_bbox is not None:
        if known_face is not None:
            from services.detection import localize_known_face_in_search_region

            search_anchor = predicted_bbox or last_bbox
            search_factor = MANUAL_FACE_SEARCH_EXPAND_FACTOR if (optical_ok or tracker_ok) else MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR
            relocked_face = localize_known_face_in_search_region(
                frame,
                known_face=known_face,
                search_bbox=expand_bbox(search_anchor, frame_w, frame_h, search_factor),
                preferred_bbox=search_anchor,
                tolerance=identity_tolerance if identity_tolerance is not None else 0.55,
                allow_geometry_fallback=known_face.get("encoding") is None,
            ) if search_anchor is not None else None

            if relocked_face is not None:
                resolved_bbox = tuple(relocked_face["bbox"])
                face_detected = True
                tracking_success = True
            else:
                # When the user selected a specific saved person, prefer briefly
                # losing the blur over letting a tracker drift onto a different
                # face that happens to cross the same area.
                resolved_bbox = None
                tracking_success = False
        else:
            search_factor = MANUAL_FACE_SEARCH_EXPAND_FACTOR if (optical_ok or tracker_ok) else MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR
            refined_bbox = detect_best_face_bbox(
                frame,
                expand_bbox(resolved_bbox, frame_w, frame_h, search_factor),
                preferred_bbox=resolved_bbox,
                allow_supplemental=(not tracker_ok or fail_count > 0),
            )
            if refined_bbox is not None:
                resolved_bbox = refined_bbox
                face_detected = True
                tracking_success = True

        if face_detected and tracker is not None and (
            periodic_reinit
            or not tracker_ok
            or bbox_iou(tracker_bbox, resolved_bbox) < 0.45
        ):
            try:
                refreshed_tracker = create_initialized_tracker(
                    small_frame,
                    expand_bbox(resolved_bbox, frame_w, frame_h, TRACKER_REINIT_BBOX_EXPAND_FACTOR),
                    scale_back,
                    scale_adaptive=True,
                )
                if refreshed_tracker is not None:
                    tracker = refreshed_tracker
                    fail_count = 0
            except (cv2.error, AttributeError, Exception):
                pass

    resolved_small_bbox = None
    if resolved_bbox is not None:
        resolved_small_bbox = frame_bbox_to_small_bbox(resolved_bbox, scale_back, small_w, small_h)

    if optical_points is not None and resolved_small_bbox is not None:
        filtered_points = []
        x1s, y1s, x2s, y2s = resolved_small_bbox
        for pt in optical_points.reshape(-1, 2):
            if x1s <= pt[0] <= x2s and y1s <= pt[1] <= y2s:
                filtered_points.append(pt)
        if len(filtered_points) >= 6:
            next_points = np.array(filtered_points, dtype=np.float32).reshape(-1, 1, 2)
        else:
            next_points = seed_tracking_points(gray_small, resolved_small_bbox)
    else:
        next_points = seed_tracking_points(gray_small, resolved_small_bbox) if resolved_small_bbox is not None else None

    display_bbox = smooth_bbox(resolved_bbox, prev_smoothed, TRACKER_SMOOTHING_ALPHA, frame_w, frame_h) if resolved_bbox is not None else (
        None if requires_identity_lock else prev_smoothed
    )

    if not tracking_success and tracker is not None and fail_count >= reinit_after_fails and last_bbox:
        try:
            reinit_factor = TRACKER_REINIT_BBOX_EXPAND_FACTOR if kind != "face" else MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR
            refreshed_tracker = create_initialized_tracker(
                small_frame,
                expand_bbox(last_bbox, frame_w, frame_h, reinit_factor),
                scale_back,
                scale_adaptive=True,
            )
            if refreshed_tracker is not None:
                tracker = refreshed_tracker
                fail_count = 0
        except (cv2.error, AttributeError, Exception):
            pass

    final_bbox = display_bbox or (None if requires_identity_lock else last_bbox)
    final_small_bbox = frame_bbox_to_small_bbox(final_bbox, scale_back, small_w, small_h) if final_bbox is not None else None
    return {
        **track,
        "tracker": tracker,
        "last_bbox": final_bbox,
        "smoothed_bbox": final_bbox,
        "small_bbox": final_small_bbox if final_small_bbox is not None else resolved_small_bbox,
        "points": next_points if final_small_bbox is not None else None,
        "fail_count": 0 if tracking_success else fail_count + 1,
    }, final_bbox


def redact_video(
    input_path,
    output_path,
    face_encodings=None,
    face_targets=None,
    object_classes=None,
    face_tolerance=None,
    obj_conf=0.25,
    blur_strength=DEFAULT_BLUR_STRENGTH,
    redaction_style="blur",
    detect_every_n=DEFAULT_DETECT_EVERY_N,
    detect_every_seconds=None,
    temporal_ranges=None,
    custom_regions=None,
    collect_custom_track_data=False,
    track_sample_fps=None,
    preview_only=False,
    progress_callback=None,
):
    face_encodings = face_encodings or []
    face_targets = face_targets or []
    object_classes = object_classes or set()
    temporal_ranges = temporal_ranges or []
    custom_regions = custom_regions or []
    prepared_custom_regions = []
    for reg in custom_regions:
        if not isinstance(reg, dict):
            continue
        try:
            anchor_sec = float(reg.get("anchor_sec", reg.get("start_sec", 0.0)) or 0.0)
        except (TypeError, ValueError):
            anchor_sec = 0.0
        prepared_custom_regions.append({**reg, "_anchor_sec": max(0.0, anchor_sec)})
    custom_regions = prepared_custom_regions

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    last_progress_stage = None
    last_progress_percent = -1

    def emit_progress(stage, progress, *, frames_processed=None, message=None):
        nonlocal last_progress_stage, last_progress_percent
        if progress_callback is None:
            return
        bounded_progress = max(0.0, min(float(progress), 1.0))
        percent = int(round(bounded_progress * 100))
        if stage == last_progress_stage and percent == last_progress_percent:
            return
        progress_callback({
            "stage": stage,
            "progress": round(bounded_progress, 4),
            "percent": percent,
            "frames_processed": int(frames_processed or 0),
            "total_frames": int(total or 0),
            "message": message,
        })
        last_progress_stage = stage
        last_progress_percent = percent

    if detect_every_seconds is not None and detect_every_seconds > 0:
        detect_every_n = max(1, int(round(fps * detect_every_seconds)))

    detect_every_n = max(1, min(detect_every_n, 10))
    emit_progress("preparing", 0.02, frames_processed=0, message="Preparing redaction job")

    logger.info("Redact: %dx%d, %.1f fps, ~%d frames, detect_every=%d, targets: %d face targets, %d face encodings, %d obj_classes, %d custom (tracked)",
                w, h, fps, total, detect_every_n, len(face_targets), len(face_encodings), len(object_classes), len(custom_regions))
    if custom_regions:
        logger.info("OpenCV version: %s (trackers require opencv-contrib-python)", cv2.__version__)
        for i, reg in enumerate(custom_regions[:3]):
            bbox = normalized_region_to_bbox(reg, w, h)
            logger.info(
                "Custom region %d: start=%.3fs norm x=%.2f y=%.2f w=%.2f h=%.2f -> pixel bbox %s",
                i,
                reg.get("_anchor_sec", 0.0),
                reg.get("x", 0),
                reg.get("y", 0),
                reg.get("width", 0),
                reg.get("height", 0),
                bbox,
            )
        if len(custom_regions) > 3:
            logger.info("... and %d more custom regions", len(custom_regions) - 3)

    temp_path = None
    writer = None
    if not preview_only:
        tmp_fd, temp_path = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(output_path) or ".")
        os.close(tmp_fd)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
    emit_progress("rendering", 0.05, frames_processed=0, message="Rendering redacted video")

    auto_face_mode = (
        isinstance(face_encodings, list)
        and len(face_encodings) == 1
        and isinstance(face_encodings[0], str)
        and face_encodings[0] == "__ALL__"
    )

    trackers = []
    frame_idx = 0
    detection_frames_processed = 0
    auto_prev_small_gray = None
    AUTO_TRACKER_REINIT_INTERVAL = 20
    AUTO_TRACKER_REINIT_AFTER_FAILS = 5

    # Motion tracking for manual (drawn) regions: tracker per region, init on frame 0, update every frame.
    # Each entry: (tracker or None for static fallback, effect, static_bbox, last_bbox from tracker or None).
    custom_trackers = [None] * len(custom_regions)  # tracker or None
    custom_started = [False] * len(custom_regions)
    custom_modes = [None] * len(custom_regions)
    custom_effects = [None] * len(custom_regions)
    custom_static_bboxes = [None] * len(custom_regions)  # initial bbox (x1,y1,x2,y2) once region becomes active
    custom_last_bboxes = [None] * len(custom_regions)    # last successful tracker bbox; used when update() fails so blur keeps following
    custom_smoothed_bboxes = [None] * len(custom_regions)  # EMA-smoothed bbox per region (when TRACKER_SMOOTHING_ALPHA > 0)
    custom_face_boxes = [None] * len(custom_regions)  # last confirmed face bbox for face-aware manual blur regions
    custom_face_paddings = [None] * len(custom_regions)  # keeps the user's drawn framing relative to the detected face
    custom_small_bboxes = [None] * len(custom_regions)  # current tracked bbox in scaled tracking frame
    custom_feature_points = [None] * len(custom_regions)  # feature points used for optical-flow tracking
    custom_fail_count = [0] * len(custom_regions)     # consecutive update() failures per region; used to re-init after several failures
    CUSTOM_TRACKER_REINIT_INTERVAL = 20   # re-init tracker every N frames so it keeps following (avoids freeze)
    CUSTOM_TRACKER_REINIT_AFTER_FAILS = 5  # re-init when update() fails this many times in a row
    sampled_custom_tracks = [
        {
            "id": str(reg.get("id", idx)),
            "shape": reg.get("shape") or "rectangle",
            "effect": reg.get("effect") or "blur",
            "anchor_sec": reg.get("_anchor_sec", 0.0),
            "samples": [],
        }
        for idx, reg in enumerate(custom_regions)
    ]
    sample_step = None
    next_sample_sec = 0.0
    if collect_custom_track_data:
        sample_step = 1.0 / max(1.0, float(track_sample_fps or fps or 1.0))
    prev_small_gray = None
    if preview_only and custom_regions:
        earliest_anchor_sec = min((float(reg.get("_anchor_sec", 0.0)) for reg in custom_regions), default=0.0)
        start_frame = max(0, min(total, int(earliest_anchor_sec * fps)))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            next_sample_sec = earliest_anchor_sec
            logger.info("Preview tracking: starting scan at %.3fs (frame %d)", earliest_anchor_sec, start_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = frame_idx / fps
        frame_custom_display_bboxes = [None] * len(custom_regions)
        in_temporal_range = True
        if temporal_ranges:
            in_temporal_range = any(
                r["start"] <= current_sec <= r["end"]
                for r in temporal_ranges
            )

        run_detection = (
            frame_idx % detect_every_n == 0
            and in_temporal_range
            and (auto_face_mode or face_targets or face_encodings or object_classes)
        )

        small = None
        gray_small = None
        scale_back_actual = None
        if custom_regions or trackers or run_detection:
            small, scale_back_actual = small_frame_for_tracking(frame, TRACKER_MAX_DIM)
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if custom_regions:
            periodic_reinit = (frame_idx % CUSTOM_TRACKER_REINIT_INTERVAL == 0)
            just_started = set()

            for idx, reg in enumerate(custom_regions):
                if custom_started[idx]:
                    continue
                if current_sec + (0.5 / max(fps, 1.0)) < reg.get("_anchor_sec", 0.0):
                    continue

                bbox = normalized_region_to_bbox(reg, w, h)
                custom_started[idx] = True
                if bbox is None:
                    continue

                effect = reg.get("effect") or "blur"
                mode = custom_region_tracking_mode(reg)
                face_bbox = None
                face_padding = None
                tracked_bbox = bbox
                if mode == "face":
                    face_bbox = detect_best_face_bbox(
                        frame,
                        expand_bbox(bbox, w, h, MANUAL_FACE_SEARCH_EXPAND_FACTOR),
                        preferred_bbox=bbox,
                        allow_supplemental=True,
                    )
                    if face_bbox:
                        face_padding = face_padding_from_bbox(bbox, face_bbox)
                        tracked_bbox = apply_face_padding(face_bbox, face_padding, w, h) or face_bbox

                frame_custom_display_bboxes[idx] = tracked_bbox
                if not preview_only:
                    apply_redaction(frame, tracked_bbox, effect, blur_strength)

                tracker = None
                try:
                    tracker = create_initialized_tracker(small, tracked_bbox, scale_back_actual, scale_adaptive=True)
                    if tracker is None:
                        logger.warning("Custom region %d tracker unavailable or init failed, using optical/static fallback.", idx)
                except (cv2.error, AttributeError, Exception) as e:
                    logger.warning("Custom region %d tracker init failed, using fallback: %s", idx, e)

                custom_trackers[idx] = tracker
                custom_modes[idx] = mode
                custom_effects[idx] = effect
                custom_static_bboxes[idx] = tracked_bbox
                custom_last_bboxes[idx] = tracked_bbox
                custom_smoothed_bboxes[idx] = tracked_bbox
                custom_face_boxes[idx] = face_bbox
                custom_face_paddings[idx] = face_padding
                custom_small_bboxes[idx] = frame_bbox_to_small_bbox(tracked_bbox, scale_back_actual, small.shape[1], small.shape[0])
                custom_feature_points[idx] = seed_tracking_points(gray_small, custom_small_bboxes[idx])
                custom_fail_count[idx] = 0
                just_started.add(idx)
                logger.info(
                    "Activated custom region %d at %.3fs (requested %.3fs, mode=%s, tracker=%s)",
                    idx,
                    current_sec,
                    reg.get("_anchor_sec", 0.0),
                    mode,
                    "yes" if tracker is not None else "no",
                )

            active_indices = [
                idx for idx, started in enumerate(custom_started)
                if started and custom_static_bboxes[idx] is not None
            ]

            for idx in active_indices:
                if idx in just_started:
                    continue

                tr = custom_trackers[idx]
                mode = custom_modes[idx] or "generic"
                static_bbox = custom_static_bboxes[idx]
                last_bbox = custom_last_bboxes[idx]
                fail_count = custom_fail_count[idx]
                prev_smoothed = custom_smoothed_bboxes[idx] if custom_smoothed_bboxes[idx] is not None else static_bbox
                prev_face_bbox = custom_face_boxes[idx]
                face_padding = custom_face_paddings[idx]
                prev_small_bbox = custom_small_bboxes[idx]
                prev_points = custom_feature_points[idx]
                use_bbox = last_bbox if last_bbox else static_bbox
                tracker_bbox = None
                tracker_ok = False
                optical_bbox = None
                optical_points = None
                optical_ok = False

                if tr is not None and periodic_reinit and use_bbox:
                    try:
                        reinit_bbox = expand_bbox(use_bbox, w, h, TRACKER_REINIT_BBOX_EXPAND_FACTOR)
                        refreshed_tracker = create_initialized_tracker(
                            small,
                            reinit_bbox,
                            scale_back_actual,
                            scale_adaptive=True,
                        )
                        if refreshed_tracker is not None:
                            tr = refreshed_tracker
                            fail_count = 0
                    except (cv2.error, AttributeError, Exception):
                        pass

                if tr is not None:
                    ok, roi = tr.update(small)
                    if ok and roi is not None:
                        try:
                            x, y, tw, th = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
                            if tw > 0 and th > 0:
                                tracker_bbox = tracker_roi_to_frame_bbox(x, y, tw, th, scale_back_actual, w, h)
                                tracker_ok = tracker_bbox is not None
                        except (IndexError, TypeError, ValueError):
                            tracker_ok = False

                if prev_small_gray is not None and prev_small_bbox is not None and prev_points is not None:
                    optical_small_bbox, optical_points = optical_flow_bbox_update(
                        prev_small_gray,
                        gray_small,
                        prev_points,
                        prev_small_bbox,
                        small.shape[1],
                        small.shape[0],
                    )
                    if optical_small_bbox is not None:
                        optical_bbox = small_bbox_to_frame_bbox(optical_small_bbox, scale_back_actual, w, h)
                        optical_ok = optical_bbox is not None

                if tr is not None and optical_ok and (not tracker_ok or bbox_iou(tracker_bbox, optical_bbox) < 0.3):
                    try:
                        optical_seed_bbox = expand_bbox(optical_bbox, w, h, TRACKER_REINIT_BBOX_EXPAND_FACTOR)
                        refreshed_tracker = create_initialized_tracker(
                            small,
                            optical_seed_bbox,
                            scale_back_actual,
                            scale_adaptive=True,
                        )
                        if refreshed_tracker is not None:
                            tr = refreshed_tracker
                            fail_count = 0
                    except (cv2.error, AttributeError, Exception):
                        pass

                resolved_bbox = optical_bbox if optical_ok else (tracker_bbox if tracker_ok else None)
                resolved_face_bbox = prev_face_bbox
                face_detected = False

                if mode == "face":
                    search_anchor = optical_bbox or tracker_bbox or prev_face_bbox or use_bbox or static_bbox
                    if search_anchor is not None:
                        search_factor = MANUAL_FACE_SEARCH_EXPAND_FACTOR if (optical_ok or tracker_ok) else MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR
                        resolved_face_bbox = detect_best_face_bbox(
                            frame,
                            expand_bbox(search_anchor, w, h, search_factor),
                            preferred_bbox=optical_bbox or tracker_bbox or prev_face_bbox or use_bbox,
                            allow_supplemental=(not tracker_ok or prev_face_bbox is None or fail_count > 0),
                        )
                    if resolved_face_bbox is not None:
                        face_detected = True
                        if face_padding is None:
                            face_padding = face_padding_from_bbox(use_bbox or static_bbox, resolved_face_bbox)
                        resolved_bbox = apply_face_padding(resolved_face_bbox, face_padding, w, h) or resolved_face_bbox
                        if tr is not None and (
                            not tracker_ok
                            or periodic_reinit
                            or bbox_iou(tracker_bbox, resolved_bbox) < 0.45
                        ):
                            try:
                                refreshed_tracker = create_initialized_tracker(
                                    small,
                                    expand_bbox(resolved_bbox, w, h, TRACKER_REINIT_BBOX_EXPAND_FACTOR),
                                    scale_back_actual,
                                    scale_adaptive=True,
                                )
                                if refreshed_tracker is not None:
                                    tr = refreshed_tracker
                                    fail_count = 0
                            except (cv2.error, AttributeError, Exception):
                                pass
                    elif tracker_ok:
                        resolved_bbox = tracker_bbox

                if resolved_bbox is None:
                    resolved_bbox = use_bbox

                resolved_small_bbox = frame_bbox_to_small_bbox(resolved_bbox, scale_back_actual, small.shape[1], small.shape[0]) if resolved_bbox is not None else prev_small_bbox
                if optical_points is not None and resolved_small_bbox is not None:
                    filtered_points = []
                    x1s, y1s, x2s, y2s = resolved_small_bbox
                    for pt in optical_points.reshape(-1, 2):
                        if x1s <= pt[0] <= x2s and y1s <= pt[1] <= y2s:
                            filtered_points.append(pt)
                    if len(filtered_points) >= 6:
                        next_points = np.array(filtered_points, dtype=np.float32).reshape(-1, 1, 2)
                    else:
                        next_points = seed_tracking_points(gray_small, resolved_small_bbox)
                else:
                    next_points = seed_tracking_points(gray_small, resolved_small_bbox)

                if resolved_bbox is not None:
                    display_bbox = smooth_bbox(resolved_bbox, prev_smoothed, TRACKER_SMOOTHING_ALPHA, w, h)
                    frame_custom_display_bboxes[idx] = display_bbox
                    if not preview_only:
                        apply_redaction(frame, display_bbox, custom_effects[idx] or "blur", blur_strength)
                    custom_last_bboxes[idx] = display_bbox
                    custom_smoothed_bboxes[idx] = display_bbox
                else:
                    frame_custom_display_bboxes[idx] = last_bbox or prev_smoothed
                    custom_last_bboxes[idx] = last_bbox
                    custom_smoothed_bboxes[idx] = prev_smoothed

                tracking_success = tracker_ok or optical_ok or face_detected
                if not tracking_success and tr is not None and fail_count >= CUSTOM_TRACKER_REINIT_AFTER_FAILS and use_bbox:
                    try:
                        reinit_factor = TRACKER_REINIT_BBOX_EXPAND_FACTOR if mode != "face" else MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR
                        refreshed_tracker = create_initialized_tracker(
                            small,
                            expand_bbox(use_bbox, w, h, reinit_factor),
                            scale_back_actual,
                            scale_adaptive=True,
                        )
                        if refreshed_tracker is not None:
                            tr = refreshed_tracker
                            fail_count = 0
                    except (cv2.error, AttributeError, Exception):
                        pass

                custom_trackers[idx] = tr
                custom_face_boxes[idx] = resolved_face_bbox
                custom_face_paddings[idx] = face_padding
                custom_small_bboxes[idx] = frame_bbox_to_small_bbox(custom_last_bboxes[idx], scale_back_actual, small.shape[1], small.shape[0]) if custom_last_bboxes[idx] is not None else resolved_small_bbox
                custom_feature_points[idx] = next_points
                custom_fail_count[idx] = 0 if tracking_success else fail_count + 1

            if frame_idx > 0 and frame_idx % 60 == 0 and active_indices:
                active = sum(1 for idx in active_indices if custom_trackers[idx] is not None)
                fails = sum(custom_fail_count[idx] for idx in active_indices)
                logger.info("Motion tracking frame %d: %d active trackers, %d total fail count", frame_idx, active, fails)
            prev_small_gray = gray_small

        if collect_custom_track_data and sample_step is not None and custom_regions:
            sample_cutoff = current_sec + (0.5 / max(fps, 1.0))
            while next_sample_sec <= sample_cutoff:
                for idx, bbox in enumerate(frame_custom_display_bboxes):
                    if bbox is None:
                        continue
                    norm = bbox_to_normalized_region(bbox, w, h)
                    if norm is None:
                        continue
                    sampled_custom_tracks[idx]["samples"].append({
                        "t": round(next_sample_sec, 4),
                        **norm,
                    })
                next_sample_sec += sample_step

        if run_detection:
            from services.detection import match_faces_in_frame, detect_face_boxes, localize_known_faces_in_frame
            localized_faces = []
            if auto_face_mode:
                face_boxes = [
                    expand_face_redaction_bbox(tuple(b["bbox"]), w, h)
                    for b in detect_face_boxes(frame)
                ]
            elif face_targets:
                localized_faces = localize_known_faces_in_frame(
                    frame,
                    face_targets,
                    time_sec=current_sec,
                    tolerance=face_tolerance if face_tolerance is not None else 0.55,
                )
                face_boxes = [
                    expand_face_redaction_bbox(tuple(face["bbox"]), w, h)
                    for face in localized_faces
                ]
            else:
                face_boxes = [
                    expand_face_redaction_bbox(tuple(box), w, h)
                    for box in (
                        match_faces_in_frame(
                            frame,
                            face_encodings,
                            tolerance=face_tolerance if face_tolerance is not None else 0.55,
                        ) if face_encodings else []
                    )
                ]
            obj_boxes = match_objects(frame, object_classes, conf_threshold=obj_conf) if object_classes else []
            detected_tracks = []
            if auto_face_mode:
                detected_tracks.extend({"kind": "face", "bbox": box} for box in face_boxes)
            elif face_targets:
                face_targets_by_person = {
                    get_face_identity(face): face
                    for face in face_targets
                    if get_face_identity(face)
                }
                for face in localized_faces:
                    expanded_bbox = expand_face_redaction_bbox(tuple(face["bbox"]), w, h)
                    person_id = get_face_identity(face)
                    detected_tracks.append({
                        "kind": "face",
                        "bbox": expanded_bbox,
                        "person_id": person_id or None,
                        "known_face": face_targets_by_person.get(person_id),
                        "identity_tolerance": face_tolerance if face_tolerance is not None else 0.55,
                    })
            else:
                detected_tracks.extend({"kind": "face", "bbox": box} for box in face_boxes)
            detected_tracks.extend({"kind": "object", "bbox": box} for box in obj_boxes)
            detection_frames_processed += 1

            for detected_track in detected_tracks:
                box = detected_track["bbox"]
                x1, y1, x2, y2 = [int(v) for v in box]
                apply_redaction(frame, (x1, y1, x2, y2), redaction_style, blur_strength)
            trackers = [
                initialize_auto_redaction_track(
                    small,
                    gray_small,
                    detected_track["bbox"],
                    scale_back_actual,
                    detected_track["kind"],
                    {
                        key: value
                        for key, value in detected_track.items()
                        if key not in {"bbox", "kind"}
                    },
                )
                for detected_track in detected_tracks
            ] if small is not None and gray_small is not None and scale_back_actual is not None else []
            auto_prev_small_gray = gray_small
        elif trackers and in_temporal_range and small is not None and gray_small is not None and scale_back_actual is not None:
            periodic_reinit = (frame_idx % AUTO_TRACKER_REINIT_INTERVAL == 0)
            next_trackers = []
            for track in trackers:
                updated_track, display_bbox = update_auto_redaction_track(
                    track,
                    frame,
                    small,
                    gray_small,
                    auto_prev_small_gray,
                    scale_back_actual,
                    w,
                    h,
                    periodic_reinit=periodic_reinit,
                    reinit_after_fails=AUTO_TRACKER_REINIT_AFTER_FAILS,
                )
                if display_bbox is not None:
                    apply_redaction(frame, display_bbox, redaction_style, blur_strength)
                if updated_track.get("last_bbox") is not None:
                    next_trackers.append(updated_track)
            trackers = next_trackers
            auto_prev_small_gray = gray_small
        else:
            if not in_temporal_range:
                trackers = []
            auto_prev_small_gray = gray_small if trackers else None

        if writer is not None:
            writer.write(frame)
        frame_idx += 1

        if total > 0:
            render_progress = 0.05 + (0.87 * min(frame_idx, total) / max(total, 1))
            emit_progress(
                "rendering",
                render_progress,
                frames_processed=frame_idx,
                message="Rendering redacted video",
            )

        if total > 0 and frame_idx % max(1, total // 10) == 0:
            pct = int(100 * frame_idx / total)
            logger.info("Redact progress: %d%% (%d/%d)", pct, frame_idx, total)

    cap.release()
    if writer is not None:
        writer.release()

    if not preview_only and temp_path is not None:
        emit_progress("reencoding", 0.94, frames_processed=frame_idx, message="Re-encoding output")
        reencode_mp4_to_h264(temp_path, output_path, original_path=input_path)
        try:
            os.remove(temp_path)
        except OSError:
            pass
        emit_progress("finalizing", 0.99, frames_processed=frame_idx, message="Finalizing output")

    logger.info(
        "%s complete: %d frames, %d detection passes%s",
        "Preview tracking" if preview_only else "Redaction",
        frame_idx,
        detection_frames_processed,
        "" if preview_only else f" -> {output_path}",
    )
    result = {
        "output_path": output_path,
        "total_frames": frame_idx,
        "fps": fps,
        "width": w,
        "height": h,
        "detection_frames_processed": detection_frames_processed,
        "detection_frames_skipped": 0,
    }
    if collect_custom_track_data:
        result["custom_tracks"] = sampled_custom_tracks
    emit_progress("completed", 1.0, frames_processed=frame_idx, message="Redaction complete")
    return result
