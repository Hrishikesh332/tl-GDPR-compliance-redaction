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
    TRACKER_SIZE_SMOOTHING_ALPHA,
    TRACKER_VELOCITY_SMOOTHING_ALPHA,
    TRACKER_PREDICTION_MAX_FRAMES,
    TRACK_ASSOCIATION_IOU,
    TRACK_ASSOCIATION_CENTER_RATIO,
    TRACK_LOST_GRACE_FRAMES,
    MANUAL_FACE_SEARCH_EXPAND_FACTOR,
    MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR,
    MANUAL_FACE_DETECTION_CONFIDENCE,
)
from services.face_identity import get_face_identity
from utils.image import apply_redaction, restore_region
from utils.video import small_frame_for_tracking, finalize_mp4_export, export_video_dimensions, normalize_export_height

logger = logging.getLogger("video_redaction.redactor")

# Face detections can flicker around the facial features, especially when the
# subject is far from camera. Export redaction favors coverage over tightness.
FACE_REDACTION_PAD_X_RATIO = 0.10
FACE_REDACTION_PAD_TOP_RATIO = 0.18
FACE_REDACTION_PAD_BOTTOM_RATIO = 0.10
GLOBAL_MOTION_MAX_CORNERS = 240
GLOBAL_MOTION_MIN_POINTS = 8
GLOBAL_MOTION_MIN_CONFIDENCE = 0.14
FACE_TRACK_MOTION_SEARCH_BONUS_CAP = 0.72
# Cap how much extra padding motion can add to the blur. Export keeps a
# modest margin so missed detector frames do not expose edges.
FACE_TRACK_MOTION_PAD_CAP = 0.14
FACE_TRACK_BRIDGE_PAD_CAP = 0.14
# Lower confidence bar so the blur can ride a camera shift even if the
# scene is mostly low-texture (background sky, walls, etc.).
FACE_TRACK_BRIDGE_MIN_GLOBAL_CONFIDENCE = 0.14
# Allow more consecutive bridged frames so brief detection blanks do not
# reveal the face on small camera shifts and quick zooms.
FACE_TRACK_BRIDGE_MAX_FAILS = 12
# Adaptive smoothing thresholds (face-bbox motion relative to face size).
# Below STILL: snap aggressively to detection so the blur sits "glued" to
# the face. Above MOVING: smooth heavily to ride out tracker noise.
FACE_LOCK_STILL_RATE = 0.012
FACE_LOCK_MOVING_RATE = 0.06
FACE_LOCK_STILL_ALPHA = 0.62
FACE_LOCK_MOVING_ALPHA = 0.36
FACE_LOCK_STILL_SIZE_ALPHA = 0.42
FACE_LOCK_MOVING_SIZE_ALPHA = 0.24
FACE_LOCK_PREDICTION_ALPHA = 0.42
FACE_LOCK_PREDICTION_SIZE_ALPHA = 0.28
FACE_LOCK_MOTION_BRIDGE_ALPHA = 0.92
FACE_LOCK_MOTION_BRIDGE_SIZE_ALPHA = 0.65
FACE_TRACK_POINT_MAX_CORNERS = 90
FACE_TRACK_POINT_INNER_RATIO = 0.9
OPTICAL_FLOW_FORWARD_BACK_MAX_ERROR = 1.8
OPTICAL_FLOW_MIN_AFFINE_INLIER_RATIO = 0.42
TEMPLATE_MATCH_MIN_SCORE = 0.46
REVERSE_FOCUS_PRESERVE_EXPAND = 2.25
REVERSE_FACE_DETECT_MAX_DIM = int(os.environ.get("REVERSE_FACE_DETECT_MAX_DIM", "960") or 960)
REVERSE_FACE_DETECT_CONFIDENCE = float(os.environ.get("REVERSE_FACE_DETECT_CONFIDENCE", "0.16") or 0.16)
REVERSE_FACE_DETECT_MIN_SIZE = int(os.environ.get("REVERSE_FACE_DETECT_MIN_SIZE", "8") or 8)
REVERSE_FACE_DETECT_MIN_SHARPNESS = float(os.environ.get("REVERSE_FACE_DETECT_MIN_SHARPNESS", "2.0") or 2.0)
REVERSE_FACE_MASK_PAD_X_RATIO = 0.04
REVERSE_FACE_MASK_PAD_TOP_RATIO = 0.05
REVERSE_FACE_MASK_PAD_BOTTOM_RATIO = 0.08
REVERSE_FOCUS_MEMORY_GRACE_FRAMES = 18
REVERSE_FOCUS_MEMORY_PAD_PER_FRAME = 0.035
REVERSE_FOCUS_MEMORY_PAD_CAP = 0.85
REVERSE_FOCUS_RESTORE_PAD_X_RATIO = 0.03
REVERSE_FOCUS_RESTORE_PAD_TOP_RATIO = 0.04
REVERSE_FOCUS_RESTORE_PAD_BOTTOM_RATIO = 0.06
NO_TRACKER_FACTORY = object()
TRACKER_FACTORY_CACHE = {}
TRACKER_FACTORY_LOGGED = set()
TRACKER_UNAVAILABLE_WARNING_EMITTED = False


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
    cached = TRACKER_FACTORY_CACHE.get(key)
    if cached is not None:
        return None if cached is NO_TRACKER_FACTORY else cached

    for name, create in tracker_factory_candidates(scale_adaptive=scale_adaptive):
        try:
            t = create()
            if t is not None:
                TRACKER_FACTORY_CACHE[key] = (name, create)
                if name not in TRACKER_FACTORY_LOGGED:
                    logger.info("Using tracker: %s for motion tracking", name)
                    TRACKER_FACTORY_LOGGED.add(name)
                return name, create
        except (AttributeError, cv2.error, Exception):
            continue

    global TRACKER_UNAVAILABLE_WARNING_EMITTED
    TRACKER_FACTORY_CACHE[key] = NO_TRACKER_FACTORY
    if not TRACKER_UNAVAILABLE_WARNING_EMITTED:
        logger.warning(
            "No preferred OpenCV tracker was available. Tracking will fall back to optical-flow/static tracking."
        )
        TRACKER_UNAVAILABLE_WARNING_EMITTED = True
    return None


def create_tracker(scale_adaptive=False):
    """Create a tracker instance using the best constructor available in this OpenCV build."""
    resolved = resolve_tracker_factory(scale_adaptive=scale_adaptive)
    if resolved is None:
        return None

    create = resolved[1]
    try:
        return create()
    except (AttributeError, cv2.error, Exception):
        # If a cached constructor later becomes unusable, clear it so future calls
        # can probe lower-priority trackers instead of retrying the same one forever.
        TRACKER_FACTORY_CACHE.pop(bool(scale_adaptive), None)
    return None


def smooth_bbox(new_bbox, prev_bbox, alpha, frame_w, frame_h, size_alpha=None):
    """Exponential moving average of a bbox at sub-pixel precision.

    ``alpha`` is the weight of the new bbox **center** (a value of 1.0 disables
    smoothing entirely). ``size_alpha`` controls smoothing of width/height
    independently — keeping it lower than ``alpha`` prevents the blur from
    pulsing during zoom while still letting it follow translation snappily.

    Floats are preserved through the EMA so that small movements between
    frames do not get rounded into pixel-level shimmer.
    """
    if alpha is None or prev_bbox is None:
        return new_bbox
    pos_alpha = max(0.0, min(1.0, float(alpha)))
    if size_alpha is None:
        sz_alpha = pos_alpha
    else:
        sz_alpha = max(0.0, min(1.0, float(size_alpha)))
    if pos_alpha >= 0.999 and sz_alpha >= 0.999:
        return new_bbox

    nx1, ny1, nx2, ny2 = new_bbox
    px1, py1, px2, py2 = prev_bbox
    new_cx = (float(nx1) + float(nx2)) / 2.0
    new_cy = (float(ny1) + float(ny2)) / 2.0
    new_w = max(1.0, float(nx2) - float(nx1))
    new_h = max(1.0, float(ny2) - float(ny1))
    prev_cx = (float(px1) + float(px2)) / 2.0
    prev_cy = (float(py1) + float(py2)) / 2.0
    prev_w = max(1.0, float(px2) - float(px1))
    prev_h = max(1.0, float(py2) - float(py1))

    cx = pos_alpha * new_cx + (1.0 - pos_alpha) * prev_cx
    cy = pos_alpha * new_cy + (1.0 - pos_alpha) * prev_cy
    w = sz_alpha * new_w + (1.0 - sz_alpha) * prev_w
    h = sz_alpha * new_h + (1.0 - sz_alpha) * prev_h

    x1 = max(0.0, min(cx - w / 2.0, float(frame_w)))
    y1 = max(0.0, min(cy - h / 2.0, float(frame_h)))
    x2 = max(0.0, min(cx + w / 2.0, float(frame_w)))
    y2 = max(0.0, min(cy + h / 2.0, float(frame_h)))
    if x2 <= x1 or y2 <= y1:
        return new_bbox
    return (x1, y1, x2, y2)


def bbox_to_state(bbox):
    """Return (cx, cy, w, h) for a bbox; None when bbox is invalid."""
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2) - float(x1))
    h = max(1.0, float(y2) - float(y1))
    cx = (float(x1) + float(x2)) / 2.0
    cy = (float(y1) + float(y2)) / 2.0
    return (cx, cy, w, h)


def state_to_bbox(state, frame_w, frame_h):
    """Convert a state tuple to a sub-pixel bbox clamped to frame bounds."""
    if not state:
        return None
    cx, cy, w, h = state
    x1 = max(0.0, min(cx - w / 2.0, float(frame_w)))
    y1 = max(0.0, min(cy - h / 2.0, float(frame_h)))
    x2 = max(0.0, min(cx + w / 2.0, float(frame_w)))
    y2 = max(0.0, min(cy + h / 2.0, float(frame_h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def face_motion_rate(velocity, state):
    """Per-frame face-bbox displacement relative to the face's own size.

    Used to decide between snappy lock-on (face is essentially still) and
    heavier smoothing (face is genuinely moving). Returns 0.0 when the
    bbox state is unknown.
    """
    if not velocity or not state:
        return 0.0
    vx, vy = velocity[:2]
    w, h = state[2:4]
    diag = max(8.0, math.hypot(w, h))
    return float(math.hypot(vx, vy) / diag)


def adaptive_lock_alpha(face_motion, scale_strength=0.0):
    """Pick (position_alpha, size_alpha) based on how fast the face is moving.

    The blur "sticks" to the face by snapping aggressively when the subject
    is essentially still (so a small camera shift does not desync), and
    smooths more when the subject is moving fast (so the blur does not jitter
    while the tracker chases the head). Scale change is also factored in so
    rapid zooms relax the size smoothing too.
    """
    motion = max(0.0, float(face_motion or 0.0))
    if motion <= FACE_LOCK_STILL_RATE:
        weight = 0.0
    elif motion >= FACE_LOCK_MOVING_RATE:
        weight = 1.0
    else:
        span = FACE_LOCK_MOVING_RATE - FACE_LOCK_STILL_RATE
        weight = (motion - FACE_LOCK_STILL_RATE) / span if span > 0 else 1.0

    pos_alpha = (
        FACE_LOCK_STILL_ALPHA * (1.0 - weight)
        + FACE_LOCK_MOVING_ALPHA * weight
    )
    size_alpha = (
        FACE_LOCK_STILL_SIZE_ALPHA * (1.0 - weight)
        + FACE_LOCK_MOVING_SIZE_ALPHA * weight
    )
    if scale_strength and scale_strength > 0.04:
        size_alpha = min(1.0, size_alpha + min(0.25, scale_strength * 2.5))
    return pos_alpha, size_alpha


def update_velocity(prev_velocity, prev_state, new_state, alpha=None):
    """Update an EMA-smoothed (vx, vy, vw, vh) velocity from two states.

    ``alpha`` is the weight of the new instantaneous velocity in the EMA
    (0..1). When the previous state is unknown, the velocity is reset to
    zero so that the predictor does not start by extrapolating from a stale
    measurement.
    """
    if new_state is None:
        return prev_velocity or (0.0, 0.0, 0.0, 0.0)
    if prev_state is None:
        return (0.0, 0.0, 0.0, 0.0)
    smoothing = TRACKER_VELOCITY_SMOOTHING_ALPHA if alpha is None else alpha
    smoothing = max(0.0, min(1.0, float(smoothing)))
    instant = (
        new_state[0] - prev_state[0],
        new_state[1] - prev_state[1],
        new_state[2] - prev_state[2],
        new_state[3] - prev_state[3],
    )
    if not prev_velocity:
        return instant
    return tuple(
        smoothing * instant[i] + (1.0 - smoothing) * prev_velocity[i]
        for i in range(4)
    )


def predict_state(prev_state, velocity, frames=1):
    """Apply ``frames`` velocity steps to ``prev_state`` to predict where the
    face will be next. Width/height are clamped to a small minimum to keep
    the bbox usable when the predictor extrapolates a long shrink streak.
    """
    if prev_state is None or velocity is None:
        return prev_state
    steps = max(0.0, float(frames))
    vx, vy, vw, vh = velocity
    cx = prev_state[0] + vx * steps
    cy = prev_state[1] + vy * steps
    w = max(8.0, prev_state[2] + vw * steps)
    h = max(8.0, prev_state[3] + vh * steps)
    return (cx, cy, w, h)


def predicted_bbox_for_track(track, frames=1):
    state = bbox_to_state(track.get("last_bbox") or track.get("smoothed_bbox"))
    return predict_state(state, track.get("velocity"), frames=frames)


def scale_change_strength(velocity, state):
    """Estimate how aggressively the face is zooming relative to its size.

    Returns 0.0 when the face is steady. Larger values are returned when
    width/height are growing or shrinking quickly so the search region and
    tracker re-init can compensate for rapid scale changes during zoom.
    """
    if not velocity or not state:
        return 0.0
    vw, vh = velocity[2:4]
    w, h = state[2:4]
    rate_w = abs(vw) / max(1.0, w)
    rate_h = abs(vh) / max(1.0, h)
    return max(rate_w, rate_h)


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


def scale_bbox_to_frame(bbox, scale_back, frame_w, frame_h):
    if not bbox:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        scale = float(scale_back or 1.0)
    except (TypeError, ValueError):
        return None
    scaled = (
        max(0, int(round(x1 * scale))),
        max(0, int(round(y1 * scale))),
        min(frame_w, int(round(x2 * scale))),
        min(frame_h, int(round(y2 * scale))),
    )
    if scaled[2] <= scaled[0] or scaled[3] <= scaled[1]:
        return None
    return scaled


def scale_frame_bbox(bbox, scale_x, scale_y, frame_w, frame_h):
    if not bbox:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        sx = float(scale_x)
        sy = float(scale_y)
    except (TypeError, ValueError):
        return None
    scaled = (
        max(0.0, min(x1 * sx, float(frame_w))),
        max(0.0, min(y1 * sy, float(frame_h))),
        max(0.0, min(x2 * sx, float(frame_w))),
        max(0.0, min(y2 * sy, float(frame_h))),
    )
    if scaled[2] <= scaled[0] or scaled[3] <= scaled[1]:
        return None
    return scaled


def scale_known_face_target_for_frame(face, scale_x, scale_y, frame_w, frame_h):
    if not isinstance(face, dict):
        return face
    scaled_face = dict(face)
    scaled_bbox = scale_frame_bbox(face.get("bbox"), scale_x, scale_y, frame_w, frame_h)
    if scaled_bbox is not None:
        scaled_face["bbox"] = [round(v, 3) for v in scaled_bbox]

    scaled_appearances = []
    for appearance in face.get("appearances") or []:
        if not isinstance(appearance, dict):
            continue
        scaled_appearance = dict(appearance)
        appearance_bbox = scale_frame_bbox(
            appearance.get("bbox"),
            scale_x,
            scale_y,
            frame_w,
            frame_h,
        )
        if appearance_bbox is not None:
            scaled_appearance["bbox"] = [round(v, 3) for v in appearance_bbox]
        scaled_appearances.append(scaled_appearance)
    if scaled_appearances:
        scaled_face["appearances"] = scaled_appearances
    return scaled_face


def detect_reverse_face_tracks(frame, frame_w, frame_h):
    """Fast all-face detector for reverse export.

    Reverse export needs every frame to move, so this path intentionally avoids
    the InsightFace whole-frame detector that is too slow for long videos on
    CPU. The tracker/refinement loop still smooths and bridges these detections
    after they are seeded.
    """
    from services.detection import detect_faces_res10, face_sharpness

    max_dim = max(360, int(REVERSE_FACE_DETECT_MAX_DIM or 960))
    detector_frame, scale_back = small_frame_for_tracking(frame, max_dim)
    det_h, det_w = detector_frame.shape[:2]
    min_face_size = max(4, int(REVERSE_FACE_DETECT_MIN_SIZE))
    boxes = detect_faces_res10(
        detector_frame,
        confidence_threshold=REVERSE_FACE_DETECT_CONFIDENCE,
        min_face_size=min_face_size,
        upscale=1.0,
    )

    tracks = []
    for x1, y1, x2, y2, conf in boxes:
        det_bbox = (
            max(0, min(int(x1), det_w)),
            max(0, min(int(y1), det_h)),
            max(0, min(int(x2), det_w)),
            max(0, min(int(y2), det_h)),
        )
        if det_bbox[2] <= det_bbox[0] or det_bbox[3] <= det_bbox[1]:
            continue
        frame_bbox = scale_bbox_to_frame(det_bbox, scale_back, frame_w, frame_h)
        if frame_bbox is None:
            continue
        if face_sharpness(frame, frame_bbox) < REVERSE_FACE_DETECT_MIN_SHARPNESS:
            continue
        tracks.append({
            "kind": "face",
            "bbox": expand_face_redaction_bbox(frame_bbox, frame_w, frame_h),
            "confidence": round(float(conf), 4),
            "source": "res10-fast",
            "fast_reverse": True,
            "disable_cv_tracker": True,
        })
    return tracks


def motion_bridge_bbox(bbox, frame_w, frame_h, motion_strength=0.0, cap=FACE_TRACK_BRIDGE_PAD_CAP):
    if not bbox:
        return bbox
    bonus = min(max(0.0, float(cap)), max(0.0, float(motion_strength or 0.0)) * 5.5)
    factor = 1.0 + bonus
    if factor <= 1.0:
        return bbox
    return expand_bbox(bbox, frame_w, frame_h, factor)


def expand_tracked_face_bbox(face_bbox, frame_w, frame_h, motion_strength=0.0):
    expanded = expand_face_redaction_bbox(face_bbox, frame_w, frame_h)
    return motion_bridge_bbox(
        expanded,
        frame_w,
        frame_h,
        motion_strength=motion_strength,
        cap=FACE_TRACK_MOTION_PAD_CAP,
    )


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


def bbox_intersection_area(box_a, box_b):
    if not box_a or not box_b:
        return 0.0
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    return float(max(0, xb - xa) * max(0, yb - ya))


def bbox_center_distance(box_a, box_b):
    if not box_a or not box_b:
        return 1e9
    acx = (box_a[0] + box_a[2]) / 2.0
    acy = (box_a[1] + box_a[3]) / 2.0
    bcx = (box_b[0] + box_b[2]) / 2.0
    bcy = (box_b[1] + box_b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def face_bbox_is_preserved(candidate_bbox, preserve_bboxes, frame_w=None, frame_h=None):
    """Return True when a detected/track face is the protected focus face.

    Reverse redaction detects every face live, so the protected identity is
    removed geometrically using the focus lane/identity-localized bbox before
    blur is applied. The checks combine IoU, containment, and center distance
    because the focus lane is intentionally padded more than raw detector boxes.
    """
    if not candidate_bbox or not preserve_bboxes:
        return False
    candidate = tuple(float(v) for v in candidate_bbox[:4])
    candidate_area = max(1.0, float(bbox_area(candidate)))
    candidate_diag = max(
        1.0,
        math.hypot(candidate[2] - candidate[0], candidate[3] - candidate[1]),
    )
    for preserve_bbox in preserve_bboxes:
        if not preserve_bbox:
            continue
        preserve = tuple(float(v) for v in preserve_bbox[:4])
        preserve_area = max(1.0, float(bbox_area(preserve)))
        preserve_diag = max(
            1.0,
            math.hypot(preserve[2] - preserve[0], preserve[3] - preserve[1]),
        )
        center_distance = bbox_center_distance(candidate, preserve)
        if center_distance <= min(candidate_diag, preserve_diag) * 0.42:
            return True
        overlap_area = bbox_intersection_area(candidate, preserve)
        iou = overlap_area / max(1.0, candidate_area + preserve_area - overlap_area)
        if iou >= 0.10:
            return True
        if overlap_area / candidate_area >= 0.42:
            return True
        if overlap_area <= 0:
            continue
        if center_distance <= max(candidate_diag * 0.52, preserve_diag * 0.42):
            return True
    return False


def filter_preserved_face_tracks(tracks, preserve_bboxes, frame_w, frame_h):
    if not preserve_bboxes:
        return tracks
    filtered = []
    for track in tracks or []:
        if track.get("kind") == "face":
            track_bbox = track.get("last_bbox") or track.get("smoothed_bbox")
            if face_bbox_is_preserved(track_bbox, preserve_bboxes, frame_w, frame_h):
                continue
        filtered.append(track)
    return filtered


def filter_reverse_focus_detected_tracks(detected_tracks, preserve_bboxes):
    if not detected_tracks or not preserve_bboxes:
        return detected_tracks

    filtered = []
    for detected_track in detected_tracks:
        if detected_track.get("kind") != "face":
            filtered.append(detected_track)
            continue
        candidate_bbox = detected_track.get("bbox")
        if not candidate_bbox:
            filtered.append(detected_track)
            continue
        candidate = tuple(float(v) for v in candidate_bbox[:4])
        candidate_area = max(1.0, float(bbox_area(candidate)))
        candidate_diag = max(
            1.0,
            math.hypot(candidate[2] - candidate[0], candidate[3] - candidate[1]),
        )
        should_remove = face_bbox_is_preserved(candidate, preserve_bboxes)
        for preserve_bbox in preserve_bboxes:
            if should_remove or not preserve_bbox:
                continue
            preserve = tuple(float(v) for v in preserve_bbox[:4])
            preserve_area = max(1.0, float(bbox_area(preserve)))
            preserve_diag = max(
                1.0,
                math.hypot(preserve[2] - preserve[0], preserve[3] - preserve[1]),
            )
            overlap = bbox_intersection_area(candidate, preserve)
            overlap_ratio = overlap / candidate_area
            preserve_overlap_ratio = overlap / preserve_area
            iou = bbox_iou(candidate, preserve)
            center_distance = bbox_center_distance(candidate, preserve)
            should_remove = (
                iou >= 0.05
                or overlap_ratio >= 0.18
                or preserve_overlap_ratio >= 0.18
                or center_distance <= max(candidate_diag * 0.65, preserve_diag * 0.55)
            )
        if not should_remove:
            filtered.append(detected_track)
    return filtered


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
        min_face_size=10,
        min_sharpness=2.0,
        upscale=2.2,
    )
    if not detections:
        try:
            from services.detection import localize_head_in_search_region

            head_match = localize_head_in_search_region(
                frame,
                search_bbox=(x1, y1, x2, y2),
                preferred_bbox=preferred_bbox,
                strict=preferred_bbox is not None,
            )
            if head_match is not None:
                return tuple(head_match["bbox"])
        except Exception:
            pass
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


def seed_tracking_points(gray_frame, bbox, max_corners=60, elliptical=False):
    if gray_frame is None or bbox is None:
        return None
    h, w = gray_frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(round(float(x1)), w - 1)))
    y1 = int(max(0, min(round(float(y1)), h - 1)))
    x2 = int(max(0, min(round(float(x2)), w)))
    y2 = int(max(0, min(round(float(y2)), h)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    if elliptical:
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        axes = (
            max(2, int(round((x2 - x1) * FACE_TRACK_POINT_INNER_RATIO / 2.0))),
            max(2, int(round((y2 - y1) * FACE_TRACK_POINT_INNER_RATIO / 2.0))),
        )
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    else:
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


def seed_track_points_for_kind(kind, gray_frame, bbox):
    """Seed per-track LK points, biasing face tracks toward the facial oval."""
    is_face = str(kind or "").strip().lower() == "face"
    points = seed_tracking_points(
        gray_frame,
        bbox,
        max_corners=FACE_TRACK_POINT_MAX_CORNERS if is_face else 60,
        elliptical=is_face,
    )
    if points is None and is_face:
        points = seed_tracking_points(gray_frame, bbox, max_corners=60, elliptical=False)
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

    next_points, status, err = cv2.calcOpticalFlowPyrLK(
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

    valid = status.flatten() == 1
    if err is not None:
        err_values = err.reshape(-1)
        valid_errors = err_values[valid]
        if valid_errors.size:
            max_err = max(12.0, float(np.median(valid_errors)) * 2.75)
            valid = valid & (err_values <= max_err)

    good_old = prev_points[valid]
    good_new = next_points[valid]
    if len(good_old) < 2 or len(good_new) < 2:
        return None, None

    # Forward-backward LK validation rejects points that drift onto the
    # background during camera pans or low-texture frames.
    back_flow_result = cv2.calcOpticalFlowPyrLK(
        gray,
        prev_gray,
        good_new,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    back_points, back_status = back_flow_result[:2]
    if back_points is not None and back_status is not None:
        fb_error = np.linalg.norm(
            back_points.reshape(-1, 2) - good_old.reshape(-1, 2),
            axis=1,
        )
        bbox_w = max(1.0, float(prev_bbox[2] - prev_bbox[0]))
        bbox_h = max(1.0, float(prev_bbox[3] - prev_bbox[1]))
        max_fb_error = max(
            OPTICAL_FLOW_FORWARD_BACK_MAX_ERROR,
            min(bbox_w, bbox_h) * 0.045,
        )
        fb_valid = (back_status.flatten() == 1) & (fb_error <= max_fb_error)
        good_old = good_old[fb_valid]
        good_new = good_new[fb_valid]
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
            inlier_ratio = 1.0
            if inliers is not None and len(inliers) > 0:
                inlier_ratio = float(np.mean(inliers))
            scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
            corners = cv2.transform(bbox_corners(prev_bbox), matrix)
            bbox = corners_to_bbox(corners, frame_w, frame_h)
            if (
                bbox is not None
                and inlier_ratio >= OPTICAL_FLOW_MIN_AFFINE_INLIER_RATIO
                and 0.55 <= scale <= 1.7
            ):
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


def template_match_bbox_update(
    prev_gray,
    gray,
    prev_bbox,
    frame_w,
    frame_h,
    *,
    search_expand=1.85,
    min_score=TEMPLATE_MATCH_MIN_SCORE,
):
    """Track a bbox by matching its previous visual patch nearby.

    This is intentionally embedding-free and detector-free. It gives the
    tracker another motion-only signal when LK points are sparse, which is
    common for far faces, backs of heads, profile turns, and blurred frames.
    """
    if prev_gray is None or gray is None or prev_bbox is None:
        return None, 0.0

    x1, y1, x2, y2 = [int(round(float(v))) for v in prev_bbox]
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w))
    y2 = max(0, min(y2, frame_h))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 8 or bh < 8:
        return None, 0.0

    template = prev_gray[y1:y2, x1:x2]
    if template.size == 0 or float(template.std()) < 3.0:
        return None, 0.0

    expand = max(1.0, float(search_expand or 1.0))
    pad_x = max(10, int(round(bw * (expand - 1.0))))
    pad_y = max(10, int(round(bh * (expand - 1.0))))
    sx1 = max(0, x1 - pad_x)
    sy1 = max(0, y1 - pad_y)
    sx2 = min(frame_w, x2 + pad_x)
    sy2 = min(frame_h, y2 + pad_y)
    if sx2 - sx1 < bw or sy2 - sy1 < bh:
        return None, 0.0

    search = gray[sy1:sy2, sx1:sx2]
    if search.size == 0:
        return None, 0.0

    try:
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        match_result = cv2.minMaxLoc(result)
        max_val = match_result[1]
        max_loc = match_result[3]
    except (cv2.error, ValueError):
        return None, 0.0

    score = float(max_val)
    if not np.isfinite(score) or score < float(min_score):
        return None, score if np.isfinite(score) else 0.0

    nx1 = sx1 + int(max_loc[0])
    ny1 = sy1 + int(max_loc[1])
    nx2 = nx1 + bw
    ny2 = ny1 + bh
    bbox = corners_to_bbox(bbox_corners((nx1, ny1, nx2, ny2)), frame_w, frame_h)
    return bbox, score


def estimate_global_frame_motion(prev_gray, gray):
    if prev_gray is None or gray is None:
        return None

    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=GLOBAL_MOTION_MAX_CORNERS,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if points is None or len(points) < GLOBAL_MOTION_MIN_POINTS:
        return None

    flow_result = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.02),
    )
    next_points, status = flow_result[:2]
    if next_points is None or status is None:
        return None

    good_old = points[status.flatten() == 1]
    good_new = next_points[status.flatten() == 1]
    if len(good_old) < GLOBAL_MOTION_MIN_POINTS or len(good_new) < GLOBAL_MOTION_MIN_POINTS:
        return None

    matrix = None
    inlier_ratio = 0.0
    residual = None
    if len(good_old) >= 10:
        matrix, inliers = cv2.estimateAffinePartial2D(
            good_old.reshape(-1, 1, 2),
            good_new.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
        )
        if matrix is not None:
            transformed = cv2.transform(good_old.reshape(-1, 1, 2), matrix).reshape(-1, 2)
            residual = float(np.median(np.linalg.norm(transformed - good_new.reshape(-1, 2), axis=1)))
            if inliers is not None and len(inliers) > 0:
                inlier_ratio = float(np.mean(inliers))

    if matrix is None:
        diff = (good_new - good_old).reshape(-1, 2)
        delta = np.median(diff, axis=0)
        dx = float(delta[0])
        dy = float(delta[1])
        residual = float(np.median(np.linalg.norm(diff - delta, axis=1)))
    else:
        dx = float(matrix[0, 2])
        dy = float(matrix[1, 2])

    frame_diag = max(1.0, float(np.hypot(gray.shape[1], gray.shape[0])))
    motion_strength = float(np.hypot(dx, dy) / frame_diag)
    feature_ratio = min(1.0, len(good_old) / GLOBAL_MOTION_MAX_CORNERS)
    residual_score = 1.0 - min(1.0, (residual or 0.0) / 12.0)
    confidence = max(0.0, min(1.0, feature_ratio * 0.45 + inlier_ratio * 0.35 + residual_score * 0.2))

    return {
        "matrix": matrix,
        "dx": dx,
        "dy": dy,
        "confidence": confidence,
        "motion_strength": motion_strength,
    }


def apply_motion_to_bbox(bbox, motion, frame_w, frame_h):
    if bbox is None or motion is None:
        return None

    matrix = motion.get("matrix")
    if matrix is not None:
        return corners_to_bbox(cv2.transform(bbox_corners(bbox), matrix), frame_w, frame_h)

    return translate_bbox(
        bbox,
        float(motion.get("dx", 0.0)),
        float(motion.get("dy", 0.0)),
        frame_w,
        frame_h,
    )


def merge_tracking_search_anchor(primary_bbox, secondary_bbox, frame_w, frame_h):
    if primary_bbox is None:
        return secondary_bbox
    if secondary_bbox is None:
        return primary_bbox
    merged_corners = np.concatenate((bbox_corners(primary_bbox), bbox_corners(secondary_bbox)), axis=0)
    return corners_to_bbox(merged_corners, frame_w, frame_h) or primary_bbox or secondary_bbox


def weighted_fuse_bboxes(candidates, frame_w, frame_h):
    """Fuse agreeing motion candidates as a small constant-velocity correction."""
    valid = [
        (bbox, max(0.0, float(weight)), name)
        for bbox, weight, name in candidates
        if bbox is not None and weight and weight > 0
    ]
    if not valid:
        return None
    anchor_bbox = max(valid, key=lambda item: item[1])[0]
    anchor_diag = max(
        1.0,
        math.hypot(anchor_bbox[2] - anchor_bbox[0], anchor_bbox[3] - anchor_bbox[1]),
    )
    clustered = []
    for bbox, weight, name in valid:
        if (
            bbox == anchor_bbox
            or bbox_iou(bbox, anchor_bbox) >= 0.18
            or bbox_center_distance(bbox, anchor_bbox) <= anchor_diag * 0.7
        ):
            clustered.append((bbox, weight, name))
    if not clustered:
        return anchor_bbox
    total = sum(item[1] for item in clustered)
    if total <= 0:
        return anchor_bbox

    cx = cy = bw = bh = 0.0
    for item in clustered:
        bbox, weight = item[0], item[1]
        state = bbox_to_state(bbox)
        if state is None:
            continue
        ratio = weight / total
        cx += state[0] * ratio
        cy += state[1] * ratio
        bw += state[2] * ratio
        bh += state[3] * ratio
    return state_to_bbox((cx, cy, bw, bh), frame_w, frame_h) or anchor_bbox


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
    metadata = metadata or {}
    tracker = None
    if not metadata.get("disable_cv_tracker"):
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
        **metadata,
        "kind": kind,
        "tracker": tracker,
        "last_bbox": bbox,
        "smoothed_bbox": bbox,
        "small_bbox": small_bbox,
        "points": seed_track_points_for_kind(kind, gray_small, small_bbox) if small_bbox is not None else None,
        "fail_count": 0,
        "velocity": (0.0, 0.0, 0.0, 0.0),
        "frames_since_detection": 0,
        "lost_streak": 0,
        "lost_count": 0,
    }


def reseed_existing_track(track, small_frame, gray_small, detection_bbox, scale_back, frame_w, frame_h):
    """Re-anchor an existing track to a fresh detection bbox without losing
    smoothing/velocity state. This keeps the blur smoothly attached to the
    same person across periodic detection passes instead of jumping every
    ``detect_every_n`` frames.
    """
    if detection_bbox is None:
        return track

    prev_state = bbox_to_state(track.get("last_bbox") or track.get("smoothed_bbox"))
    velocity = track.get("velocity") or (0.0, 0.0, 0.0, 0.0)
    kind = str(track.get("kind") or "object").strip().lower()

    if kind == "face":
        # Use the same adaptive lock as the per-frame update so a fresh
        # detection on a still face snaps tightly while a fresh detection
        # on a moving face still smooths. This eliminates the jump that
        # used to happen on every detection pass.
        face_motion = face_motion_rate(velocity, prev_state)
        scale_strength = scale_change_strength(velocity, prev_state) if prev_state else 0.0
        pos_alpha, size_alpha = adaptive_lock_alpha(face_motion, scale_strength)
    else:
        pos_alpha = TRACKER_SMOOTHING_ALPHA
        size_alpha = TRACKER_SIZE_SMOOTHING_ALPHA

    smoothed = smooth_bbox(
        detection_bbox,
        track.get("smoothed_bbox") or track.get("last_bbox") or detection_bbox,
        pos_alpha,
        frame_w,
        frame_h,
        size_alpha=size_alpha,
    ) or detection_bbox

    new_state = bbox_to_state(detection_bbox)
    velocity = update_velocity(velocity, prev_state, new_state)

    tracker = track.get("tracker")
    if not track.get("disable_cv_tracker"):
        try:
            refreshed = create_initialized_tracker(
                small_frame,
                expand_bbox(detection_bbox, frame_w, frame_h, TRACKER_REINIT_BBOX_EXPAND_FACTOR),
                scale_back,
                scale_adaptive=True,
            )
            if refreshed is not None:
                tracker = refreshed
        except (cv2.error, AttributeError, Exception):
            pass

    small_bbox = frame_bbox_to_small_bbox(
        detection_bbox,
        scale_back,
        small_frame.shape[1],
        small_frame.shape[0],
    )

    new_track = dict(track)
    new_track.update({
        "tracker": tracker,
        "last_bbox": smoothed,
        "smoothed_bbox": smoothed,
        "small_bbox": small_bbox,
        "points": seed_track_points_for_kind(kind, gray_small, small_bbox) if small_bbox is not None else None,
        "velocity": velocity,
        "fail_count": 0,
        "frames_since_detection": 0,
        "lost_streak": 0,
        "lost_count": 0,
    })
    return new_track


def associate_detections_to_tracks(detected_tracks, existing_tracks, frame_w, frame_h):
    """Match fresh detection bboxes to existing track ids, preserving each
    track's tracker, smoothing buffer, and identity across detection passes.

    For face tracks tied to a known person, person_id is matched first so the
    correct identity stays locked even if the bbox briefly shifts. For other
    tracks, we use IoU and bbox-center proximity (relative to the track's
    diagonal) so a fresh detection is associated with whichever existing
    track is closest, preventing the tracker from being torn down and
    rebuilt every ``detect_every_n`` frames.
    """
    matches = {}
    used_existing = set()
    used_detections = set()

    if not existing_tracks or not detected_tracks:
        return matches, set(range(len(detected_tracks))), set(range(len(existing_tracks)))

    candidates = []
    for d_idx, detection in enumerate(detected_tracks):
        det_bbox = detection.get("bbox")
        det_kind = detection.get("kind")
        det_person = detection.get("person_id")
        for e_idx, track in enumerate(existing_tracks):
            if track.get("kind") != det_kind:
                continue
            track_bbox = track.get("last_bbox") or track.get("smoothed_bbox")
            if track_bbox is None or det_bbox is None:
                continue

            track_person = track.get("person_id")
            identity_match = bool(det_person and track_person and det_person == track_person)
            if (det_person or track_person) and not identity_match:
                if det_person and track_person:
                    continue

            iou = bbox_iou(det_bbox, track_bbox)
            track_diag = max(
                1.0,
                math.hypot(track_bbox[2] - track_bbox[0], track_bbox[3] - track_bbox[1]),
            )
            center_ratio = bbox_center_distance(det_bbox, track_bbox) / track_diag
            if not identity_match:
                if iou < TRACK_ASSOCIATION_IOU and center_ratio > TRACK_ASSOCIATION_CENTER_RATIO:
                    continue

            score = (3.0 if identity_match else 0.0) + iou * 2.5 - center_ratio * 1.0
            candidates.append((score, d_idx, e_idx, identity_match))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for candidate in candidates:
        d_idx, e_idx = candidate[1], candidate[2]
        if d_idx in used_detections or e_idx in used_existing:
            continue
        matches[d_idx] = e_idx
        used_detections.add(d_idx)
        used_existing.add(e_idx)

    unmatched_detections = set(range(len(detected_tracks))) - used_detections
    unmatched_existing = set(range(len(existing_tracks))) - used_existing
    return matches, unmatched_detections, unmatched_existing


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
    velocity = track.get("velocity") or (0.0, 0.0, 0.0, 0.0)
    frames_since_detection = int(track.get("frames_since_detection", 0) or 0)
    lost_streak = int(track.get("lost_streak", 0) or 0)
    small_h, small_w = small_frame.shape[:2]

    tracker_bbox = None
    tracker_ok = False
    optical_bbox = None
    optical_points = None
    optical_ok = False
    template_bbox = None
    template_score = 0.0
    template_ok = False
    global_motion = estimate_global_frame_motion(prev_gray_small, gray_small) if prev_gray_small is not None else None
    global_motion_confidence = float((global_motion or {}).get("confidence", 0.0) or 0.0)
    motion_strength = float((global_motion or {}).get("motion_strength", 0.0) or 0.0)
    global_bbox = None
    if global_motion is not None and global_motion_confidence >= GLOBAL_MOTION_MIN_CONFIDENCE:
        global_bbox = apply_motion_to_bbox(last_bbox or prev_smoothed, global_motion, frame_w, frame_h)

    last_state = bbox_to_state(last_bbox or prev_smoothed)
    predicted_state = predict_state(last_state, velocity, frames=1) if last_state else None
    predicted_motion_bbox = state_to_bbox(predicted_state, frame_w, frame_h) if predicted_state else None
    scale_strength = scale_change_strength(velocity, last_state) if last_state else 0.0

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

    if prev_gray_small is not None and prev_small_bbox is not None:
        template_small_bbox, template_score = template_match_bbox_update(
            prev_gray_small,
            gray_small,
            prev_small_bbox,
            small_w,
            small_h,
            search_expand=1.95 if kind == "face" else 1.7,
        )
        if template_small_bbox is not None:
            template_bbox = small_bbox_to_frame_bbox(template_small_bbox, scale_back, frame_w, frame_h)
            template_ok = template_bbox is not None

    motion_reinit_bbox = optical_bbox or template_bbox
    if tracker is not None and motion_reinit_bbox is not None and (
        not tracker_ok or bbox_iou(tracker_bbox, motion_reinit_bbox) < 0.3
    ):
        try:
            refreshed_tracker = create_initialized_tracker(
                small_frame,
                expand_bbox(motion_reinit_bbox, frame_w, frame_h, TRACKER_REINIT_BBOX_EXPAND_FACTOR),
                scale_back,
                scale_adaptive=True,
            )
            if refreshed_tracker is not None:
                tracker = refreshed_tracker
                fail_count = 0
        except (cv2.error, AttributeError, Exception):
            pass

    has_measured_motion = optical_ok or template_ok or tracker_ok or global_bbox is not None
    predicted_bbox = weighted_fuse_bboxes(
        [
            (optical_bbox, 3.2 if optical_ok else 0.0, "optical"),
            (template_bbox, 2.0 * max(0.5, template_score) if template_ok else 0.0, "template"),
            (tracker_bbox, 2.2 if tracker_ok else 0.0, "tracker"),
            (
                global_bbox,
                1.1 + global_motion_confidence if global_bbox is not None else 0.0,
                "global",
            ),
            (
                predicted_motion_bbox,
                0.8 if (
                    not has_measured_motion
                    and predicted_motion_bbox is not None
                    and frames_since_detection < TRACKER_PREDICTION_MAX_FRAMES
                ) else 0.0,
                "velocity",
            ),
        ],
        frame_w,
        frame_h,
    )
    resolved_bbox = predicted_bbox or last_bbox
    face_detected = False
    head_detected = False
    tracking_success = tracker_ok or optical_ok or template_ok or global_bbox is not None

    # Search bonus combines whole-frame motion (camera shift) and per-track
    # scale velocity (face zooming in/out). This keeps the face in view
    # while the tracker catches up to a sudden zoom or pan.
    search_motion_bonus = min(
        FACE_TRACK_MOTION_SEARCH_BONUS_CAP,
        motion_strength * 6.0 + scale_strength * 2.4,
    )

    if kind == "face" and resolved_bbox is not None:
        if known_face is not None:
            from services.detection import localize_known_face_in_search_region

            search_anchor = merge_tracking_search_anchor(predicted_bbox or last_bbox, global_bbox, frame_w, frame_h)
            search_anchor = merge_tracking_search_anchor(search_anchor, predicted_motion_bbox, frame_w, frame_h)
            search_anchor = merge_tracking_search_anchor(search_anchor, template_bbox, frame_w, frame_h)
            search_factor = MANUAL_FACE_SEARCH_EXPAND_FACTOR if (optical_ok or template_ok or tracker_ok or global_bbox is not None) else MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR
            search_factor += search_motion_bonus
            # Keep the relock bounded to the tracked neighborhood, but allow
            # the same strict geometry fallback used during initial anchored
            # localization so deploy/runtime embedding drift does not blank out
            # a selected-face blur lane.
            relocked_face = localize_known_face_in_search_region(
                frame,
                known_face=known_face,
                search_bbox=expand_bbox(search_anchor, frame_w, frame_h, search_factor),
                preferred_bbox=predicted_bbox or predicted_motion_bbox or global_bbox or last_bbox,
                tolerance=identity_tolerance if identity_tolerance is not None else 0.55,
                allow_geometry_fallback=True,
            ) if search_anchor is not None else None

            if relocked_face is not None:
                resolved_bbox = expand_tracked_face_bbox(tuple(relocked_face["bbox"]), frame_w, frame_h, motion_strength)
                face_detected = True
                tracking_success = True
            else:
                try:
                    from services.detection import localize_head_in_search_region

                    head_match = localize_head_in_search_region(
                        frame,
                        search_bbox=expand_bbox(search_anchor, frame_w, frame_h, search_factor),
                        preferred_bbox=predicted_bbox or predicted_motion_bbox or global_bbox or last_bbox,
                        strict=True,
                    ) if search_anchor is not None else None
                except Exception:
                    head_match = None

                if head_match is not None:
                    resolved_bbox = expand_tracked_face_bbox(tuple(head_match["bbox"]), frame_w, frame_h, motion_strength)
                    head_detected = True
                    tracking_success = True
                elif (predicted_bbox or predicted_motion_bbox) is not None and frames_since_detection < TRACKER_PREDICTION_MAX_FRAMES:
                    # Hold the blur on the predicted location while detection
                    # briefly misses (very common during fast pans / zooms).
                    resolved_bbox = motion_bridge_bbox(predicted_bbox or predicted_motion_bbox, frame_w, frame_h, motion_strength)
                    tracking_success = True
                elif global_bbox is not None and global_motion_confidence >= FACE_TRACK_BRIDGE_MIN_GLOBAL_CONFIDENCE and fail_count < FACE_TRACK_BRIDGE_MAX_FAILS:
                    # During a hard camera pan, a selected face can miss for a frame
                    # even though the whole scene motion is clear. Bridge with the
                    # globally translated box instead of briefly revealing the face.
                    resolved_bbox = motion_bridge_bbox(global_bbox, frame_w, frame_h, motion_strength)
                    tracking_success = True
                else:
                    # When the user selected a specific saved person, prefer briefly
                    # losing the blur over letting a tracker drift onto a different
                    # face that happens to cross the same area.
                    resolved_bbox = None
                    tracking_success = False
        else:
            search_anchor = merge_tracking_search_anchor(resolved_bbox, global_bbox, frame_w, frame_h)
            search_anchor = merge_tracking_search_anchor(search_anchor, predicted_motion_bbox, frame_w, frame_h)
            search_anchor = merge_tracking_search_anchor(search_anchor, template_bbox, frame_w, frame_h)
            search_factor = MANUAL_FACE_SEARCH_EXPAND_FACTOR if (optical_ok or template_ok or tracker_ok or global_bbox is not None) else MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR
            search_factor += search_motion_bonus
            if track.get("fast_reverse"):
                refined_bbox = None
            else:
                refined_bbox = detect_best_face_bbox(
                    frame,
                    expand_bbox(search_anchor, frame_w, frame_h, search_factor),
                    preferred_bbox=predicted_bbox or predicted_motion_bbox or global_bbox or resolved_bbox,
                    allow_supplemental=(not tracker_ok or fail_count > 0),
                )
            if refined_bbox is not None:
                resolved_bbox = expand_tracked_face_bbox(refined_bbox, frame_w, frame_h, motion_strength)
                face_detected = True
                tracking_success = True
            elif (predicted_bbox or predicted_motion_bbox) is not None and frames_since_detection < TRACKER_PREDICTION_MAX_FRAMES:
                resolved_bbox = motion_bridge_bbox(predicted_bbox or predicted_motion_bbox, frame_w, frame_h, motion_strength)
                tracking_success = True
            elif global_bbox is not None and global_motion_confidence >= FACE_TRACK_BRIDGE_MIN_GLOBAL_CONFIDENCE:
                resolved_bbox = motion_bridge_bbox(global_bbox, frame_w, frame_h, motion_strength)
                tracking_success = True

        # When zoom is changing rapidly, the on-frame tracker can lag the
        # actual face size; aggressively re-init with the latest detection
        # so the tracker keeps the box snapped to the head.
        scale_drifting = scale_strength > 0.04

        if (face_detected or head_detected) and tracker is not None and (
            periodic_reinit
            or not tracker_ok
            or bbox_iou(tracker_bbox, resolved_bbox) < 0.45
            or scale_drifting
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
            next_points = seed_track_points_for_kind(kind, gray_small, resolved_small_bbox)
    else:
        next_points = seed_track_points_for_kind(kind, gray_small, resolved_small_bbox) if resolved_small_bbox is not None else None

    if resolved_bbox is not None:
        if kind == "face":
            face_motion = face_motion_rate(velocity, last_state)
            if face_detected or head_detected:
                # Snap aggressively when the face is essentially still so a
                # tiny camera shift cannot desync the blur, but smooth more
                # when the face is genuinely moving so the blur does not
                # jitter while the tracker chases the head.
                pos_alpha, size_alpha = adaptive_lock_alpha(face_motion, scale_strength)
            elif optical_ok or template_ok or global_bbox is not None:
                # Camera-motion bridges should move the blur with the frame,
                # not lazily chase it, otherwise a fast pan can expose an edge.
                pos_alpha = FACE_LOCK_MOTION_BRIDGE_ALPHA
                size_alpha = FACE_LOCK_MOTION_BRIDGE_SIZE_ALPHA
            else:
                # Pure prediction / tracker updates — bias toward smoother
                # transitions so the blur does not drift on noisy frames.
                pos_alpha = FACE_LOCK_PREDICTION_ALPHA
                size_alpha = FACE_LOCK_PREDICTION_SIZE_ALPHA
        else:
            pos_alpha = TRACKER_SMOOTHING_ALPHA if face_detected else max(TRACKER_SMOOTHING_ALPHA * 0.65, 0.25)
            size_alpha = TRACKER_SIZE_SMOOTHING_ALPHA if face_detected else max(TRACKER_SIZE_SMOOTHING_ALPHA * 0.65, 0.18)
        display_bbox = smooth_bbox(
            resolved_bbox,
            prev_smoothed,
            pos_alpha,
            frame_w,
            frame_h,
            size_alpha=size_alpha,
        )
    elif requires_identity_lock:
        display_bbox = None
    else:
        display_bbox = prev_smoothed

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

    new_state = bbox_to_state(final_bbox)
    new_velocity = update_velocity(velocity, last_state, new_state)
    if face_detected or head_detected:
        new_frames_since_detection = 0
        new_lost_streak = 0
    else:
        new_frames_since_detection = frames_since_detection + 1
        new_lost_streak = lost_streak + (0 if tracking_success else 1)

    return {
        **track,
        "tracker": tracker,
        "last_bbox": final_bbox,
        "smoothed_bbox": final_bbox,
        "small_bbox": final_small_bbox if final_small_bbox is not None else resolved_small_bbox,
        "points": next_points if final_small_bbox is not None else None,
        "fail_count": 0 if tracking_success else fail_count + 1,
        "velocity": new_velocity,
        "frames_since_detection": new_frames_since_detection,
        "lost_streak": new_lost_streak,
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
    output_height=720,
    progress_callback=None,
    face_lock_tracks=None,
    reverse_face_redaction=False,
    preserve_face_targets=None,
    preserve_face_lock_tracks=None,
):
    face_encodings = face_encodings or []
    face_targets = face_targets or []
    preserve_face_targets = preserve_face_targets or []
    object_classes = object_classes or set()
    temporal_ranges = temporal_ranges or []
    custom_regions = custom_regions or []
    if reverse_face_redaction:
        face_encodings = ["__ALL__"]
        face_targets = []
        object_classes = set()

    # Face-lock lanes are precomputed per-frame tracks for selected people.
    # When available, they draw the blur directly so export does not depend on
    # frame-by-frame identity relock during camera shake, zoom, or profile turns.
    face_lock_tracks = face_lock_tracks or {}
    face_lock_lanes_by_person = {}
    face_lock_bboxes_by_frame = {}
    if face_lock_tracks:
        for pid, lane_doc in face_lock_tracks.items():
            if not lane_doc or not isinstance(lane_doc, dict):
                continue
            lane_array = lane_doc.get("lane") or []
            if not lane_array:
                continue
            face_lock_lanes_by_person[str(pid)] = lane_doc
            for entry in lane_array:
                try:
                    f = int(entry.get("f"))
                except (TypeError, ValueError):
                    continue
                bbox = (
                    float(entry.get("x1", 0.0)),
                    float(entry.get("y1", 0.0)),
                    float(entry.get("x2", 0.0)),
                    float(entry.get("y2", 0.0)),
                )
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                face_lock_bboxes_by_frame.setdefault(f, []).append({
                    "person_id": str(pid),
                    "bbox": bbox,
                })

    if face_lock_lanes_by_person and face_targets:
        # Remove lane-backed people from the live relock list; otherwise the
        # renderer can double-process the same face with two competing boxes.
        remaining_targets = [
            face for face in face_targets
            if str(get_face_identity(face) or "") not in face_lock_lanes_by_person
        ]
        if len(remaining_targets) != len(face_targets):
            face_targets = remaining_targets
            face_encodings = [
                np.array(face["encoding"]) if not isinstance(face.get("encoding"), np.ndarray) else face["encoding"]
                for face in face_targets
                if face.get("encoding") is not None
            ]
    prepared_custom_regions = []

    preserve_face_lock_tracks = preserve_face_lock_tracks or {}
    preserve_lock_bboxes_by_frame = {}
    if preserve_face_lock_tracks:
        for pid, lane_doc in preserve_face_lock_tracks.items():
            if not lane_doc or not isinstance(lane_doc, dict):
                continue
            for entry in lane_doc.get("lane") or []:
                try:
                    f = int(entry.get("f"))
                    bbox = (
                        float(entry.get("x1", 0.0)),
                        float(entry.get("y1", 0.0)),
                        float(entry.get("x2", 0.0)),
                        float(entry.get("y2", 0.0)),
                    )
                except (TypeError, ValueError):
                    continue
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                preserve_lock_bboxes_by_frame.setdefault(f, []).append({
                    "person_id": str(pid),
                    "bbox": bbox,
                })

    for reg in custom_regions:
        if not isinstance(reg, dict):
            continue
        try:
            anchor_sec = float(reg.get("anchor_sec", reg.get("start_sec", 0.0)) or 0.0)
        except (TypeError, ValueError):
            anchor_sec = 0.0
        prepared_custom_regions.append({**reg, "resolved_anchor_sec": max(0.0, anchor_sec)})
    custom_regions = prepared_custom_regions

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_w = w
    source_h = h
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    normalized_output_height = normalize_export_height(output_height)
    if preview_only:
        output_w, output_h = w, h
    else:
        output_w, output_h = export_video_dimensions(w, h, normalized_output_height)
    process_at_output_resolution = (
        not preview_only
        and reverse_face_redaction
        and output_w > 0
        and output_h > 0
        and (output_w != source_w or output_h != source_h)
        and (not preserve_face_targets or bool(preserve_lock_bboxes_by_frame))
    )
    active_preserve_face_targets = preserve_face_targets
    if process_at_output_resolution:
        scale_x = output_w / float(source_w)
        scale_y = output_h / float(source_h)
        for entries in preserve_lock_bboxes_by_frame.values():
            for entry in entries:
                bbox = entry.get("bbox")
                if not bbox:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                entry["bbox"] = (
                    max(0.0, min(x1 * scale_x, float(output_w))),
                    max(0.0, min(y1 * scale_y, float(output_h))),
                    max(0.0, min(x2 * scale_x, float(output_w))),
                    max(0.0, min(y2 * scale_y, float(output_h))),
                )
        active_preserve_face_targets = [
            scale_known_face_target_for_frame(face, scale_x, scale_y, output_w, output_h)
            for face in preserve_face_targets
        ]
        w, h = output_w, output_h

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

    logger.info("Redact: %dx%d -> %dx%d (%dp), %.1f fps, ~%d frames, detect_every=%d, mode=%s, targets: %d face targets, %d face encodings, %d preserve faces, %d obj_classes, %d custom (tracked)",
                source_w, source_h, output_w, output_h, normalized_output_height, fps, total, detect_every_n, "reverse_faces" if reverse_face_redaction else "standard", len(face_targets), len(face_encodings), len(preserve_face_targets), len(object_classes), len(custom_regions))
    if custom_regions:
        logger.info("OpenCV version: %s (trackers require opencv-contrib-python)", cv2.__version__)
        for i, reg in enumerate(custom_regions[:3]):
            bbox = normalized_region_to_bbox(reg, w, h)
            logger.info(
                "Custom region %d: start=%.3fs norm x=%.2f y=%.2f w=%.2f h=%.2f -> pixel bbox %s",
                i,
                reg.get("resolved_anchor_sec", 0.0),
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
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (output_w, output_h))
        if not writer.isOpened():
            raise ValueError(f"Cannot create redacted MP4 writer: {temp_path}")
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
    reverse_preserve_frames = 0
    reverse_preserve_lane_frames = 0
    reverse_preserve_localized_frames = 0
    reverse_preserve_memory_frames = 0
    reverse_focus_memory_bboxes = []
    reverse_focus_memory_frame_idx = None
    auto_prev_small_gray = None
    AUTO_TRACKER_REINIT_INTERVAL = 20
    AUTO_TRACKER_REINIT_AFTER_FAILS = 5

    custom_trackers = [None] * len(custom_regions)  # tracker or None
    custom_started = [False] * len(custom_regions)
    custom_modes = [None] * len(custom_regions)
    custom_effects = [None] * len(custom_regions)
    custom_static_bboxes = [None] * len(custom_regions)
    custom_last_bboxes = [None] * len(custom_regions)
    custom_smoothed_bboxes = [None] * len(custom_regions)
    custom_face_boxes = [None] * len(custom_regions)
    custom_face_paddings = [None] * len(custom_regions)
    custom_small_bboxes = [None] * len(custom_regions)
    custom_feature_points = [None] * len(custom_regions)
    custom_fail_count = [0] * len(custom_regions)
    CUSTOM_TRACKER_REINIT_INTERVAL = 20
    CUSTOM_TRACKER_REINIT_AFTER_FAILS = 5
    sampled_custom_tracks = [
        {
            "id": str(reg.get("id", idx)),
            "shape": reg.get("shape") or "rectangle",
            "effect": reg.get("effect") or "blur",
            "anchor_sec": reg.get("resolved_anchor_sec", 0.0),
            "samples": [],
        }
        for idx, reg in enumerate(custom_regions)
    ]
    sample_step = None
    next_sample_sec = 0.0
    if collect_custom_track_data:
        sample_step = 1.0 / max(1.0, float(track_sample_fps or fps or 1.0))
    prev_small_gray = None

    def remember_reverse_focus_bboxes(focus_bboxes):
        nonlocal reverse_focus_memory_bboxes, reverse_focus_memory_frame_idx
        valid_bboxes = [
            tuple(float(v) for v in bbox[:4])
            for bbox in focus_bboxes or []
            if bbox and bbox[2] > bbox[0] and bbox[3] > bbox[1]
        ]
        if not valid_bboxes:
            return
        reverse_focus_memory_bboxes = valid_bboxes
        reverse_focus_memory_frame_idx = frame_idx

    def reverse_focus_bboxes_from_memory():
        if reverse_focus_memory_frame_idx is None or not reverse_focus_memory_bboxes:
            return []
        age = frame_idx - reverse_focus_memory_frame_idx
        if age < 0 or age > REVERSE_FOCUS_MEMORY_GRACE_FRAMES:
            return []
        pad = min(
            REVERSE_FOCUS_MEMORY_PAD_CAP,
            REVERSE_FOCUS_MEMORY_PAD_PER_FRAME * max(1, age),
        )
        return [
            expand_bbox(bbox, w, h, 1.0 + pad)
            for bbox in reverse_focus_memory_bboxes
        ]

    def apply_detection_redaction(target_frame, bbox, kind="face"):
        kind_normalized = str(kind or "").lower()
        shape = "rect"
        redaction_bbox = bbox
        if kind_normalized == "face":
            if reverse_face_redaction and face_bbox_is_preserved(bbox, current_preserve_bboxes, w, h):
                return target_frame
            if reverse_face_redaction:
                redaction_bbox = expand_face_redaction_bbox(
                    bbox,
                    w,
                    h,
                    pad_x_ratio=REVERSE_FACE_MASK_PAD_X_RATIO,
                    pad_top_ratio=REVERSE_FACE_MASK_PAD_TOP_RATIO,
                    pad_bottom_ratio=REVERSE_FACE_MASK_PAD_BOTTOM_RATIO,
                )
                if face_bbox_is_preserved(redaction_bbox, current_preserve_bboxes, w, h):
                    return target_frame
        return apply_redaction(
            target_frame,
            redaction_bbox,
            redaction_style,
            blur_strength,
            shape=shape,
        )

    if preview_only and custom_regions:
        earliest_anchor_sec = min((float(reg.get("resolved_anchor_sec", 0.0)) for reg in custom_regions), default=0.0)
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
        if process_at_output_resolution:
            frame = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA)
        reverse_focus_restore_frame = (
            frame.copy()
            if reverse_face_redaction and (active_preserve_face_targets or preserve_lock_bboxes_by_frame)
            else None
        )

        current_sec = frame_idx / fps
        current_preserve_bboxes = [
            entry["bbox"]
            for entry in preserve_lock_bboxes_by_frame.get(frame_idx, ())
            if entry.get("bbox")
        ]
        had_lane_preserve_bbox = bool(current_preserve_bboxes)
        if face_lock_bboxes_by_frame and not preview_only:
            for entry in face_lock_bboxes_by_frame.get(frame_idx, ()):  # iterates 0..N
                lane_bbox = entry.get("bbox")
                if lane_bbox:
                    apply_detection_redaction(frame, lane_bbox, "face")
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
            and (reverse_face_redaction or auto_face_mode or face_targets or face_encodings or object_classes)
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
                if current_sec + (0.5 / max(fps, 1.0)) < reg.get("resolved_anchor_sec", 0.0):
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
                custom_feature_points[idx] = seed_track_points_for_kind(mode, gray_small, custom_small_bboxes[idx])
                custom_fail_count[idx] = 0
                just_started.add(idx)
                logger.info(
                    "Activated custom region %d at %.3fs (requested %.3fs, mode=%s, tracker=%s)",
                    idx,
                    current_sec,
                    reg.get("resolved_anchor_sec", 0.0),
                    mode,
                    "yes" if tracker is not None else "no",
                )

            active_indices = [
                idx for idx, started in enumerate(custom_started)
                if started and custom_static_bboxes[idx] is not None
            ]
            custom_global_motion = estimate_global_frame_motion(prev_small_gray, gray_small) if prev_small_gray is not None else None
            custom_global_confidence = float((custom_global_motion or {}).get("confidence", 0.0) or 0.0)

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
                global_bbox = None
                if custom_global_motion is not None and custom_global_confidence >= GLOBAL_MOTION_MIN_CONFIDENCE:
                    global_bbox = apply_motion_to_bbox(use_bbox, custom_global_motion, w, h)

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

                resolved_bbox = weighted_fuse_bboxes(
                    [
                        (optical_bbox, 3.0 if optical_ok else 0.0, "optical"),
                        (tracker_bbox, 2.0 if tracker_ok else 0.0, "tracker"),
                        (
                            global_bbox,
                            1.0 + custom_global_confidence if global_bbox is not None else 0.0,
                            "global",
                        ),
                    ],
                    w,
                    h,
                )
                resolved_face_bbox = prev_face_bbox
                face_detected = False

                if mode == "face":
                    search_anchor = merge_tracking_search_anchor(resolved_bbox, global_bbox, w, h)
                    search_anchor = search_anchor or optical_bbox or tracker_bbox or prev_face_bbox or use_bbox or static_bbox
                    if search_anchor is not None:
                        search_factor = MANUAL_FACE_SEARCH_EXPAND_FACTOR if (optical_ok or tracker_ok or global_bbox is not None) else MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR
                        resolved_face_bbox = detect_best_face_bbox(
                            frame,
                            expand_bbox(search_anchor, w, h, search_factor),
                            preferred_bbox=resolved_bbox or optical_bbox or tracker_bbox or global_bbox or prev_face_bbox or use_bbox,
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
                    elif global_bbox is not None:
                        resolved_bbox = global_bbox

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
                        next_points = seed_track_points_for_kind(mode, gray_small, resolved_small_bbox)
                else:
                    next_points = seed_track_points_for_kind(mode, gray_small, resolved_small_bbox)

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

                tracking_success = tracker_ok or optical_ok or face_detected or global_bbox is not None
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
            localized_preserve_count = 0
            if reverse_face_redaction and active_preserve_face_targets and not current_preserve_bboxes:
                preserved_faces = localize_known_faces_in_frame(
                    frame,
                    active_preserve_face_targets,
                    time_sec=current_sec,
                    tolerance=face_tolerance if face_tolerance is not None else 0.55,
                )
                for face in preserved_faces:
                    bbox = face.get("bbox")
                    if bbox:
                        preserve_bbox = expand_face_redaction_bbox(tuple(bbox), w, h)
                        current_preserve_bboxes.append(preserve_bbox)
                        localized_preserve_count += 1
            if reverse_face_redaction and current_preserve_bboxes:
                remember_reverse_focus_bboxes(current_preserve_bboxes)
            elif reverse_face_redaction and (active_preserve_face_targets or preserve_lock_bboxes_by_frame):
                memory_bboxes = reverse_focus_bboxes_from_memory()
                if memory_bboxes:
                    current_preserve_bboxes.extend(memory_bboxes)
                    reverse_preserve_memory_frames += 1
            if reverse_face_redaction and current_preserve_bboxes:
                reverse_preserve_frames += 1
                if had_lane_preserve_bbox:
                    reverse_preserve_lane_frames += 1
                elif localized_preserve_count:
                    reverse_preserve_localized_frames += 1
            localized_faces = []
            if reverse_face_redaction and auto_face_mode:
                face_boxes = []
                detected_tracks = detect_reverse_face_tracks(frame, w, h)
            elif auto_face_mode:
                face_boxes = [
                    expand_face_redaction_bbox(tuple(b["bbox"]), w, h)
                    for b in detect_face_boxes(
                        frame,
                        confidence_threshold=0.16,
                        include_supplemental=True,
                        min_face_size=10,
                        min_sharpness=2.0,
                        upscale=2.2,
                    )
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
            if not (reverse_face_redaction and auto_face_mode):
                detected_tracks = []
            if auto_face_mode and not (reverse_face_redaction and auto_face_mode):
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
            if reverse_face_redaction and current_preserve_bboxes:
                detected_tracks = filter_reverse_focus_detected_tracks(detected_tracks, current_preserve_bboxes)
                trackers = filter_preserved_face_tracks(trackers, current_preserve_bboxes, w, h)
            detection_frames_processed += 1

            if small is not None and gray_small is not None and scale_back_actual is not None:
                matches, unmatched_detections, unmatched_existing = associate_detections_to_tracks(
                    detected_tracks,
                    trackers,
                    w,
                    h,
                )

                refreshed_trackers = []
                # Matched detections reseed existing tracks while preserving
                # smoothing and velocity, so the visible mask does not jump to
                # a raw detector box on every detection pass.
                drawn_detection_indices = set()

                for d_idx, e_idx in matches.items():
                    detection = detected_tracks[d_idx]
                    existing = trackers[e_idx]
                    metadata = {
                        key: value
                        for key, value in detection.items()
                        if key not in {"bbox", "kind"}
                    }
                    merged_track = {**existing, **metadata}
                    reseeded = reseed_existing_track(
                        merged_track,
                        small,
                        gray_small,
                        detection["bbox"],
                        scale_back_actual,
                        w,
                        h,
                    )
                    refreshed_trackers.append(reseeded)
                    seed_bbox = reseeded.get("smoothed_bbox") or reseeded.get("last_bbox") or detection["bbox"]
                    apply_detection_redaction(frame, seed_bbox, detection.get("kind"))
                    drawn_detection_indices.add(d_idx)

                for d_idx in sorted(unmatched_detections):
                    detection = detected_tracks[d_idx]
                    metadata = {
                        key: value
                        for key, value in detection.items()
                        if key not in {"bbox", "kind"}
                    }
                    refreshed_trackers.append(initialize_auto_redaction_track(
                        small,
                        gray_small,
                        detection["bbox"],
                        scale_back_actual,
                        detection["kind"],
                        metadata,
                    ))
                    if d_idx not in drawn_detection_indices:
                        apply_detection_redaction(frame, detection["bbox"], detection.get("kind"))
                        drawn_detection_indices.add(d_idx)

                # Bridge short detector misses with the track's own motion.
                # Identity-locked faces avoid generic fallback detections so a
                # selected person's blur cannot drift onto a nearby face.
                for e_idx in sorted(unmatched_existing):
                    existing = trackers[e_idx]
                    requires_identity = existing.get("kind") == "face" and existing.get("known_face") is not None
                    lost_count = int(existing.get("lost_count", 0) or 0) + 1
                    if lost_count > TRACK_LOST_GRACE_FRAMES:
                        continue

                    periodic_reinit = (frame_idx % AUTO_TRACKER_REINIT_INTERVAL == 0)
                    refreshed, bridge_bbox = update_auto_redaction_track(
                        existing,
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
                    refreshed["lost_count"] = lost_count

                    if bridge_bbox is None and not requires_identity:
                        bridge_state = bbox_to_state(refreshed.get("smoothed_bbox") or refreshed.get("last_bbox"))
                        if bridge_state is not None:
                            bridge_state = predict_state(
                                bridge_state,
                                refreshed.get("velocity") or (0.0, 0.0, 0.0, 0.0),
                                frames=1,
                            )
                            bridge_bbox = (
                                state_to_bbox(bridge_state, w, h)
                                or refreshed.get("smoothed_bbox")
                                or refreshed.get("last_bbox")
                            )
                        else:
                            bridge_bbox = refreshed.get("smoothed_bbox") or refreshed.get("last_bbox")

                    if bridge_bbox is not None:
                        apply_detection_redaction(frame, bridge_bbox, refreshed.get("kind"))
                        refreshed_trackers.append(refreshed)

                # If association skipped a detection for any reason, still draw
                # it once so a newly visible face is never left uncovered.
                for d_idx, detected_track in enumerate(detected_tracks):
                    if d_idx in drawn_detection_indices:
                        continue
                    apply_detection_redaction(frame, detected_track["bbox"], detected_track.get("kind"))

                trackers = refreshed_trackers
            else:
                for detected_track in detected_tracks:
                    apply_detection_redaction(frame, detected_track["bbox"], detected_track.get("kind"))
                trackers = []
            auto_prev_small_gray = gray_small
        elif trackers and in_temporal_range and small is not None and gray_small is not None and scale_back_actual is not None:
            if reverse_face_redaction and not current_preserve_bboxes:
                memory_bboxes = reverse_focus_bboxes_from_memory()
                if memory_bboxes:
                    current_preserve_bboxes.extend(memory_bboxes)
                    reverse_preserve_memory_frames += 1
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
                if reverse_face_redaction and face_bbox_is_preserved(display_bbox, current_preserve_bboxes, w, h):
                    continue
                if display_bbox is not None:
                    apply_detection_redaction(frame, display_bbox, updated_track.get("kind"))
                if updated_track.get("last_bbox") is None:
                    continue
                if int(updated_track.get("lost_streak", 0) or 0) > TRACK_LOST_GRACE_FRAMES:
                    continue
                next_trackers.append(updated_track)
            trackers = next_trackers
            auto_prev_small_gray = gray_small
        else:
            if not in_temporal_range:
                trackers = []
            auto_prev_small_gray = gray_small if trackers else None

        if reverse_focus_restore_frame is not None and current_preserve_bboxes:
            for focus_bbox in current_preserve_bboxes:
                restore_bbox = expand_face_redaction_bbox(
                    focus_bbox,
                    w,
                    h,
                    pad_x_ratio=REVERSE_FOCUS_RESTORE_PAD_X_RATIO,
                    pad_top_ratio=REVERSE_FOCUS_RESTORE_PAD_TOP_RATIO,
                    pad_bottom_ratio=REVERSE_FOCUS_RESTORE_PAD_BOTTOM_RATIO,
                )
                restore_region(frame, reverse_focus_restore_frame, restore_bbox)

        if writer is not None:
            if output_w != w or output_h != h:
                frame_to_write = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA)
            else:
                frame_to_write = frame
            writer.write(frame_to_write)
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

    output_metadata = None
    h264_encoded = False
    if not preview_only and temp_path is not None:
        emit_progress("reencoding", 0.94, frames_processed=frame_idx, message="Re-encoding output")
        output_metadata = finalize_mp4_export(temp_path, output_path, original_path=input_path)
        h264_encoded = output_metadata["h264_encoded"]
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
        "width": output_w,
        "height": output_h,
        "source_width": source_w,
        "source_height": source_h,
        "output_height": output_h,
        "export_quality": f"{normalized_output_height}p",
        "detection_frames_processed": detection_frames_processed,
        "detection_frames_skipped": 0,
    }
    if reverse_face_redaction:
        result["reverse_preserve_frames"] = reverse_preserve_frames
        result["reverse_preserve_lane_frames"] = reverse_preserve_lane_frames
        result["reverse_preserve_localized_frames"] = reverse_preserve_localized_frames
        result["reverse_preserve_memory_frames"] = reverse_preserve_memory_frames
        result["reverse_preserve_target_count"] = len(preserve_face_targets)
    if output_metadata is not None:
        result["output_size_bytes"] = output_metadata["size_bytes"]
        result["h264_encoded"] = h264_encoded
        result["download_ready"] = True
    if collect_custom_track_data:
        result["custom_tracks"] = sampled_custom_tracks
    emit_progress("completed", 1.0, frames_processed=frame_idx, message="Redaction complete")
    return result
