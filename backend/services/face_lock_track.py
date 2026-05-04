import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from config import SNAPS_DIR
from services import twelvelabs_service
from services.face_identity import get_face_identity
from services.redactor import (
    adaptive_lock_alpha,
    apply_motion_to_bbox,
    bbox_center_distance,
    bbox_iou,
    bbox_to_state,
    create_initialized_tracker,
    estimate_global_frame_motion,
    expand_face_redaction_bbox,
    face_motion_rate,
    frame_bbox_to_small_bbox,
    optical_flow_bbox_update,
    scale_change_strength,
    seed_track_points_for_kind,
    small_bbox_to_frame_bbox,
    smooth_bbox,
    template_match_bbox_update,
    update_velocity,
    weighted_fuse_bboxes,
)
from utils.video import small_frame_for_tracking

logger = logging.getLogger("video_redaction.face_lock_track")

LANE_BUILD_VERSION = 6
FACE_LOCK_TRACKS_DIRNAME = "face_lock_tracks"
DEFAULT_TRACKER_MAX_DIM = 960

# Lane bboxes are persisted slightly larger than the raw detector bbox so the
# face-shaped export mask never exposes forehead/chin edges during motion.
FACE_LOCK_SAFETY_PAD_RATIO = 0.035
FACE_LOCK_FAR_HEAD_PAD_RATIO = 0.065
FACE_LOCK_HEAD_FALLBACK_PAD_RATIO = 0.025

# Sparse appearance anchors are grouped into segments, then widened in time so
# the lane covers the whole visible presence instead of only sampled frames.
APPEARANCE_SEGMENT_MAX_GAP_SEC = 4.0
SEGMENT_TIME_PADDING_SEC = 0.9

# Identity verification corrects drift, but visual motion remains the main
# position signal unless the tracker is weak or clearly lost.
IDENTITY_VERIFY_INTERVAL_FRAMES = 2
HEAD_FALLBACK_INTERVAL_FRAMES = 8
IDENTITY_VERIFY_SEARCH_EXPAND = 2.25
IDENTITY_VERIFY_MIN_SIMILARITY = 0.34
IDENTITY_VERIFY_HARD_LOCK_SIMILARITY = 0.45
IDENTITY_VERIFY_MAX_SNAP_RATIO = 1.5

# CSRT can grow onto nearby textures; LK points usually preserve tighter face
# scale, so large area disagreement reduces the tracker weight.
TRACKER_SCALE_DISAGREEMENT_RATIO = 0.25
MOTION_VERIFY_AGREE_IOU = 0.28
MOTION_VERIFY_AGREE_CENTER_RATIO = 0.52
MOTION_VERIFY_SOFT_BLEND = 0.28
MOTION_VERIFY_RECOVERY_BLEND = 0.78
MOTION_SCALE_STEP_CAP = 0.075

lane_build_states = {}
lane_build_lock = threading.Lock()


def face_lock_lane_dir(job_id):
    return os.path.join(SNAPS_DIR, job_id, FACE_LOCK_TRACKS_DIRNAME)


def face_lock_lane_path(job_id, person_id):
    safe_pid = str(person_id or "").strip().replace(os.sep, "_")
    if not safe_pid:
        safe_pid = "unknown"
    return os.path.join(face_lock_lane_dir(job_id), f"{safe_pid}.json")


def get_face_lock_lane(job_id, person_id):
    """Return the cached lane dict if a usable, current-version lane file
    is on disk.

    The lane format and build algorithm are versioned via
    ``LANE_BUILD_VERSION``. Any persisted lane whose ``build_version``
    is older than the running build version is treated as stale and
    discarded, forcing the caller to rebuild it under the current
    algorithm. This is critical because the v1 -> v2 upgrade introduced
    dense InsightFace verification, scale stabilization and
    bidirectional smoothing — reusing a v1 lane gives objectively
    worse tracking than a fresh v2 build, even though the JSON would
    deserialize fine. Lanes built under the current version (or a
    forward-compatible newer version) are returned as-is.
    """
    path = face_lock_lane_path(job_id, person_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not load cached face-lock lane at %s", path, exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    lane = data.get("lane")
    if not isinstance(lane, list) or not lane:
        return None
    try:
        cached_version = int(data.get("build_version") or 0)
    except (TypeError, ValueError):
        cached_version = 0
    if cached_version < LANE_BUILD_VERSION:
        logger.info(
            "Discarding stale face-lock lane at %s (build_version=%d, "
            "current=%d); a fresh build will run on next request so the "
            "tracker uses the current algorithm.",
            path, cached_version, LANE_BUILD_VERSION,
        )
        return None
    return data


def set_build_state(job_id, person_id, **patch):
    key = f"{job_id}:{person_id}"
    with lane_build_lock:
        existing = lane_build_states.get(key) or {}
        existing.update(patch)
        lane_build_states[key] = existing


def get_face_lock_build_status(job_id, person_id):
    """Return the latest build status dict for a (job, person) pair."""
    key = f"{job_id}:{person_id}"
    with lane_build_lock:
        existing = lane_build_states.get(key)
        if existing:
            return dict(existing)
    if get_face_lock_lane(job_id, person_id) is not None:
        return {"status": "ready", "progress": 1.0, "percent": 100}
    return {"status": "missing", "progress": 0.0, "percent": 0}


def normalize_appearance_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def collect_person_appearances(face):
    out = []
    for app in face.get("appearances") or []:
        if not isinstance(app, dict):
            continue
        try:
            timestamp = float(app.get("timestamp"))
        except (TypeError, ValueError):
            continue
        bbox = normalize_appearance_bbox(app.get("bbox"))
        if bbox is None:
            continue
        try:
            frame_idx = int(app.get("frame_idx"))
        except (TypeError, ValueError):
            frame_idx = None
        out.append({
            "timestamp": float(timestamp),
            "frame_idx": frame_idx,
            "bbox": bbox,
        })
    base_bbox = normalize_appearance_bbox(face.get("bbox"))
    if not out and base_bbox is not None:
        try:
            base_ts = float(face.get("timestamp"))
        except (TypeError, ValueError):
            base_ts = None
        if base_ts is not None:
            out.append({"timestamp": base_ts, "frame_idx": None, "bbox": base_bbox})
    out.sort(key=lambda item: item["timestamp"])
    return out


def get_entity_search_ranges(face, video_id):
    entity_id = str(face.get("entity_id") or "").strip()
    if not entity_id or not video_id:
        return []
    try:
        ranges = twelvelabs_service.entity_search_time_ranges(
            entity_id=entity_id, video_id=video_id,
        )
    except Exception as exc:
        logger.warning("TwelveLabs entity search failed for %s: %s", entity_id, exc)
        return []
    return [
        (float(r.get("start", 0.0)), float(r.get("end", 0.0)))
        for r in ranges or []
        if isinstance(r, dict)
    ]


def build_face_lock_segments(appearances, entity_ranges, fps, total_frames, duration_sec):
    """Group anchor appearances into contiguous segments and assign frame ranges.

    A segment starts at the earliest anchor (or TwelveLabs window start)
    and runs through the latest contiguous anchor. Anchors more than
    ``APPEARANCE_SEGMENT_MAX_GAP_SEC`` apart split into a fresh segment so
    the visual tracker is restarted at a known good location after a cut.
    """
    if fps <= 0:
        fps = 25.0
    if not appearances and not entity_ranges:
        return []

    grouped = []
    current = []
    last_ts = None
    for app in appearances:
        ts = app["timestamp"]
        if last_ts is None or (ts - last_ts) <= APPEARANCE_SEGMENT_MAX_GAP_SEC:
            current.append(app)
        else:
            grouped.append(current)
            current = [app]
        last_ts = ts
    if current:
        grouped.append(current)

    segments = []
    for group in grouped:
        if not group:
            continue
        group_start = group[0]["timestamp"]
        group_end = group[-1]["timestamp"]
        # Extend with any TwelveLabs range that overlaps this anchor group
        # so coverage tracks the underlying scene rather than the sparse
        # ~1 Hz appearance grid.
        ext_start = group_start
        ext_end = group_end
        for r_start, r_end in entity_ranges or []:
            if r_end < group_start - APPEARANCE_SEGMENT_MAX_GAP_SEC:
                continue
            if r_start > group_end + APPEARANCE_SEGMENT_MAX_GAP_SEC:
                continue
            ext_start = min(ext_start, r_start)
            ext_end = max(ext_end, r_end)
        ext_start = max(0.0, ext_start - SEGMENT_TIME_PADDING_SEC)
        if duration_sec and duration_sec > 0:
            ext_end = min(duration_sec, ext_end + SEGMENT_TIME_PADDING_SEC)
        else:
            ext_end = ext_end + SEGMENT_TIME_PADDING_SEC

        start_frame = max(0, int(round(ext_start * fps)))
        end_frame = max(start_frame, int(round(ext_end * fps)))
        if total_frames and total_frames > 0:
            end_frame = min(end_frame, total_frames - 1)
        segments.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "anchors": group,
        })

    # Merge adjacent segments that overlap after padding.
    merged = []
    for seg in sorted(segments, key=lambda s: s["start_frame"]):
        if merged and seg["start_frame"] <= merged[-1]["end_frame"] + 1:
            merged[-1]["end_frame"] = max(merged[-1]["end_frame"], seg["end_frame"])
            merged[-1]["anchors"].extend(seg["anchors"])
        else:
            merged.append(seg)

    for seg in merged:
        seg["anchors"].sort(key=lambda a: a["timestamp"])

    return merged


def appearance_frame_index(app, fps):
    if app.get("frame_idx") is not None:
        try:
            return int(app["frame_idx"])
        except (TypeError, ValueError):
            pass
    try:
        return max(0, int(round(float(app["timestamp"]) * fps)))
    except (TypeError, ValueError):
        return 0


def read_next_frame(cap):
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return frame


def seek_and_read_frame(cap, target_frame, *, max_advance=240):
    """Seek to ``target_frame`` and read until the capture position
    matches it. Returns the BGR frame at ``target_frame`` or None on
    failure. ``max_advance`` caps the post-seek advance so we never
    walk forever on broken streams.
    """
    target = max(0, int(target_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    last_frame = None
    for attempt in range(max_advance + 1):
        pos_before = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        last_frame = frame
        if pos_before >= target:
            return frame
    return last_frame


# JPEG quality used when buffering small frames for the backward
# tracking pass. 92 keeps tracker-relevant detail (face contours, eye
# corners for LK seed points) without blowing the memory budget; for
# a 640x360 small frame this lands at ~30-50 KB per frame, so a
# 1000-frame segment costs ~50 MB instead of the ~700 MB an
# uncompressed buffer would need.
BACKWARD_BUFFER_JPEG_QUALITY = 92


def read_segment_small_frame_buffer(
    cap,
    start_frame,
    end_frame,
    *,
    jpeg_quality=BACKWARD_BUFFER_JPEG_QUALITY,
    max_dim=DEFAULT_TRACKER_MAX_DIM,
):
    """Read frames in ``[start_frame, end_frame]`` sequentially
    (one cap.set + N cap.read calls) and return ``(buffer, scale_back)``.

    ``buffer`` is a dict mapping ``frame_idx -> jpeg_bytes`` for the
    downscaled tracking frame at each index. ``scale_back`` is the
    common down/up-scale factor applied during ``small_frame_for_tracking``
    (constant across the segment because frame dimensions don't change).

    Used by the backward direction of ``track_segment_one_direction``
    so it can walk the segment in reverse without re-issuing
    ``cap.set(POS_FRAMES, ...)`` per step. On H.264-encoded video each
    such backward seek rewinds to the previous keyframe (~95 frames
    on a typical GOP) and decodes forward to the target — measured at
    ~7.7 fps on this video versus ~805 fps for sequential reads, a
    ~100x slowdown that turns long segments into multi-minute stalls.

    On a read failure the buffer ends at the last successful frame;
    callers are expected to break their loop when ``buffer.get(idx)``
    returns ``None``.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    buffer = {}
    scale_back = None
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    pos = start_frame
    while pos <= end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        small_frame, sb = small_frame_for_tracking(frame, max_dim=max_dim)
        if scale_back is None:
            scale_back = sb
        ok, jpg = cv2.imencode(".jpg", small_frame, encode_params)
        if not ok:
            break
        buffer[pos] = jpg.tobytes()
        pos += 1
    return buffer, scale_back


def decode_buffered_small_frame(jpg_bytes):
    """Decode a JPEG-encoded small frame back to a BGR ndarray.

    Returns ``None`` if ``jpg_bytes`` is falsy or the decode fails,
    so callers can treat it as the same kind of "frame missing"
    sentinel that ``cap.read()`` produces.
    """
    if not jpg_bytes:
        return None
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bbox_diagonal(bbox):
    if not bbox:
        return 0.0
    x1, y1, x2, y2 = bbox
    return float(np.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1)))


def bbox_center(bbox):
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_size(bbox):
    if not bbox:
        return (0.0, 0.0)
    x1, y1, x2, y2 = bbox
    return (max(0.0, x2 - x1), max(0.0, y2 - y1))


def expand_search_bbox(bbox, expand, frame_w, frame_h):
    """Expand a face bbox into a wider search region for re-detection."""
    if not bbox:
        return None
    cx, cy = bbox_center(bbox)
    w, h = bbox_size(bbox)
    if w <= 0 or h <= 0:
        return None
    half_w = w * expand * 0.5
    half_h = h * expand * 0.5
    return (
        max(0.0, cx - half_w),
        max(0.0, cy - half_h),
        min(float(frame_w), cx + half_w),
        min(float(frame_h), cy + half_h),
    )


def verify_face_at_frame(full_frame, tracked_bbox, known_face, frame_w, frame_h):
    """Run InsightFace inside a search region around ``tracked_bbox`` and
    return the matching face's bbox if found.

    Returns ``(bbox, similarity)`` or ``None``. Importing the detection
    helper inside the function avoids paying the InsightFace startup
    cost when a build path doesn't actually need verification.
    """
    if full_frame is None or tracked_bbox is None or known_face is None:
        return None
    search_bbox = expand_search_bbox(
        tracked_bbox, IDENTITY_VERIFY_SEARCH_EXPAND, frame_w, frame_h,
    )
    if search_bbox is None:
        return None
    try:
        from services.detection import localize_known_face_in_search_region
    except ImportError:
        return None

    try:
        match = localize_known_face_in_search_region(
            full_frame,
            known_face,
            search_bbox,
            preferred_bbox=tracked_bbox,
            tolerance=0.55,
            allow_geometry_fallback=False,
        )
    except Exception:
        return None
    if match is None or match.get("bbox") is None:
        return None
    similarity = float(match.get("match_score") or 0.0)
    if similarity < IDENTITY_VERIFY_MIN_SIMILARITY:
        return None
    bbox = tuple(float(v) for v in match["bbox"])
    if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None

    # Reject obvious teleports: a "snap" that moves the bbox more than
    # its own diagonal almost certainly grabbed a different face on
    # screen. Better to keep the smooth track than to jump.
    diag = bbox_diagonal(tracked_bbox)
    if diag > 0:
        cx_t, cy_t = bbox_center(tracked_bbox)
        cx_m, cy_m = bbox_center(bbox)
        if math.hypot(cx_m - cx_t, cy_m - cy_t) > diag * IDENTITY_VERIFY_MAX_SNAP_RATIO:
            return None

    return bbox, similarity


def scale_disagreement_penalty(tracker_bbox, optical_bbox):
    """Return a multiplier on the CSRT weight when CSRT and LK disagree
    on bbox area. CSRT can grow when adjacent textures match (eg. a
    coat collar near the chin); LK is purely point-based and tends to
    keep tighter scale on faces.
    """
    if tracker_bbox is None or optical_bbox is None:
        return 1.0
    tw, th = bbox_size(tracker_bbox)
    ow, oh = bbox_size(optical_bbox)
    if tw <= 0 or th <= 0 or ow <= 0 or oh <= 0:
        return 1.0
    track_area = tw * th
    optical_area = ow * oh
    larger = max(track_area, optical_area)
    smaller = min(track_area, optical_area)
    if smaller <= 0:
        return 0.5
    ratio = (larger / smaller) - 1.0
    if ratio <= TRACKER_SCALE_DISAGREEMENT_RATIO:
        return 1.0
    # Linearly fall off to 0.4 weight as disagreement grows past 100%.
    return max(0.4, 1.0 - min(1.0, (ratio - TRACKER_SCALE_DISAGREEMENT_RATIO) / 0.75) * 0.6)


def clamp_bbox_scale_step(candidate_bbox, reference_bbox, frame_w, frame_h, *, max_step=MOTION_SCALE_STEP_CAP):
    """Limit single-frame bbox scale jumps while preserving motion center.

    Detector bboxes and CSRT can pulse in size on profile turns or far
    heads. This keeps scale changes physically plausible frame-to-frame
    while still allowing smooth zoom through accumulated motion.
    """
    if candidate_bbox is None or reference_bbox is None:
        return candidate_bbox
    cc = bbox_center(candidate_bbox)
    if cc is None:
        return candidate_bbox
    cw, ch = bbox_size(candidate_bbox)
    rw, rh = bbox_size(reference_bbox)
    if cw <= 0 or ch <= 0 or rw <= 0 or rh <= 0:
        return candidate_bbox

    growth = 1.0 + max(0.0, float(max_step or 0.0))
    shrink = 1.0 / growth
    clamped_w = min(max(cw, rw * shrink), rw * growth)
    clamped_h = min(max(ch, rh * shrink), rh * growth)
    if abs(clamped_w - cw) < 0.001 and abs(clamped_h - ch) < 0.001:
        return candidate_bbox

    cx, cy = cc
    x1 = max(0.0, cx - clamped_w / 2.0)
    y1 = max(0.0, cy - clamped_h / 2.0)
    x2 = min(float(frame_w), cx + clamped_w / 2.0)
    y2 = min(float(frame_h), cy + clamped_h / 2.0)
    if x2 <= x1 or y2 <= y1:
        return candidate_bbox
    return (x1, y1, x2, y2)


def verification_agrees_with_motion(verified_bbox, motion_bbox):
    if verified_bbox is None or motion_bbox is None:
        return False, 1.0, 0.0
    diag = max(1.0, bbox_diagonal(motion_bbox))
    center_ratio = bbox_center_distance(verified_bbox, motion_bbox) / diag
    iou = bbox_iou(verified_bbox, motion_bbox)
    agrees = iou >= MOTION_VERIFY_AGREE_IOU or center_ratio <= MOTION_VERIFY_AGREE_CENTER_RATIO
    return agrees, center_ratio, iou


def choose_anchor_candidate(candidates, reference_bbox=None):
    """Choose one anchor bbox from same-frame candidates.

    Identity clustering can occasionally produce multiple bboxes for the
    same selected person at the same timestamp. During tracking, prefer
    the candidate that best continues the current motion path.
    """
    if not candidates:
        return None
    if len(candidates) == 1 or reference_bbox is None:
        return candidates[0]

    ref_diag = max(1.0, bbox_diagonal(reference_bbox))
    ref_area = max(1.0, bbox_size(reference_bbox)[0] * bbox_size(reference_bbox)[1])
    best = None
    best_score = -1e9
    for candidate in candidates:
        bbox = candidate.get("bbox")
        if bbox is None:
            continue
        cand_area = max(1.0, bbox_size(bbox)[0] * bbox_size(bbox)[1])
        center_ratio = bbox_center_distance(bbox, reference_bbox) / ref_diag
        area_penalty = abs(math.log(cand_area / ref_area))
        score = bbox_iou(bbox, reference_bbox) * 3.2 - center_ratio * 1.8 - area_penalty * 0.35
        if score > best_score:
            best = candidate
            best_score = score
    return best or candidates[0]


def choose_seed_anchor(anchors_by_frame, seed_frame, direction, reference_bbox=None):
    candidates = anchors_by_frame.get(seed_frame) or []
    if len(candidates) <= 1:
        return candidates[0] if candidates else None

    neighbor_frames = sorted(
        f for f in anchors_by_frame.keys()
        if (f > seed_frame if direction > 0 else f < seed_frame)
    )
    if direction < 0:
        neighbor_frames.reverse()

    best = None
    best_score = -1e9
    for candidate in candidates:
        bbox = candidate.get("bbox")
        if bbox is None:
            continue
        score = 0.0
        comparisons = 0
        if reference_bbox is not None:
            ref_diag = max(1.0, bbox_diagonal(reference_bbox))
            ref_area = max(1.0, bbox_size(reference_bbox)[0] * bbox_size(reference_bbox)[1])
            cand_area = max(1.0, bbox_size(bbox)[0] * bbox_size(bbox)[1])
            ref_center_ratio = bbox_center_distance(bbox, reference_bbox) / ref_diag
            ref_area_penalty = abs(math.log(cand_area / ref_area))
            score += (
                bbox_iou(bbox, reference_bbox) * 4.0
                - ref_center_ratio * 1.55
                - ref_area_penalty * 0.3
            )
            comparisons += 1
        for neighbor_frame in neighbor_frames[:3]:
            neighbor = choose_anchor_candidate(anchors_by_frame.get(neighbor_frame) or [], bbox)
            if neighbor is None:
                continue
            neighbor_bbox = neighbor.get("bbox")
            if neighbor_bbox is None:
                continue
            diag = max(1.0, bbox_diagonal(bbox))
            score += bbox_iou(bbox, neighbor_bbox) * 2.6
            score -= bbox_center_distance(bbox, neighbor_bbox) / diag
            comparisons += 1
        if comparisons == 0:
            score = -candidates.index(candidate) * 0.001
        else:
            score /= comparisons
        if score > best_score:
            best = candidate
            best_score = score
    return best or candidates[0]


def track_segment_one_direction(
    cap,
    direction,
    anchor_bbox_initial,
    start_frame,
    end_frame,
    frame_w,
    frame_h,
    anchors_by_frame,
    *,
    on_frame=None,
    known_face=None,
    verify_interval_frames=IDENTITY_VERIFY_INTERVAL_FRAMES,
):
    """Walk a segment in ``direction`` ('forward' or 'backward') from
    ``start_frame`` to ``end_frame`` (inclusive) and emit a per-frame
    bbox using the visual tracker fusion (CSRT + Lucas-Kanade + global
    motion) plus anchor pinning.

    Every ``verify_interval_frames`` frames the tracked bbox is
    additionally checked against an InsightFace detection inside a
    search region around it. Verification confirms identity and nudges
    agreeing motion tracks; it only hard-recovers the bbox when the
    visual track is weak or lost.

    Returns a dict {frame_idx: {"bbox": (x1,y1,x2,y2), "src": str, "conf": float}}.
    """
    results = {}
    if start_frame > end_frame:
        return results

    step = 1 if direction == "forward" else -1
    if direction == "forward":
        cur = start_frame
        last = end_frame
    else:
        cur = end_frame
        last = start_frame

    # Backward direction: pre-buffer all small frames in the segment by
    # reading sequentially forward, then iterate the buffer in reverse.
    # Per-step backward seeking via cap.set rewinds to the previous
    # H.264 keyframe (~95 frames on this video's GOP) and re-decodes
    # forward, which on long segments turns the loop into O(N²) decode
    # work and is the dominant cause of "stuck at 87%" stalls. A single
    # forward sweep of the segment costs O(N) reads at sequential read
    # speed (~100x faster than per-frame seeks measured on this video).
    # Verification is disabled in backward because (a) we only have the
    # downscaled small frame buffered, not the full frame InsightFace
    # needs, and (b) the overlap region [first_anc, last_anc] is
    # already covered by forward verification, so backward verification
    # was redundant there; only the small pre-anchor window
    # [seg_start, first_anc-1] (~15 frames per segment) loses
    # verification, which the bidirectional smoothing pass and the
    # tracker fusion mask out.
    backward_small_buffer = None
    backward_scale_back = None
    if direction == "backward":
        backward_small_buffer, backward_scale_back = read_segment_small_frame_buffer(
            cap, start_frame, end_frame,
        )
        if not backward_small_buffer:
            return results
        verify_interval_frames = 0

    if direction == "backward":
        small_frame = decode_buffered_small_frame(backward_small_buffer.get(cur))
        if small_frame is None:
            return results
        scale_back = backward_scale_back
        frame = None
    else:
        frame = seek_and_read_frame(cap, cur)
        if frame is None:
            return results
        small_frame, scale_back = small_frame_for_tracking(
            frame, max_dim=DEFAULT_TRACKER_MAX_DIM,
        )
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    init_bbox = anchor_bbox_initial
    tracker = None
    try:
        tracker = create_initialized_tracker(small_frame, init_bbox, scale_back, scale_adaptive=True)
    except Exception:
        tracker = None

    small_bbox = frame_bbox_to_small_bbox(init_bbox, scale_back, small_frame.shape[1], small_frame.shape[0])
    points = seed_track_points_for_kind("face", gray_small, small_bbox) if small_bbox is not None else None
    smoothed_bbox = init_bbox
    last_bbox = init_bbox
    velocity = (0.0, 0.0, 0.0, 0.0)
    prev_gray_small = gray_small
    prev_small_bbox = small_bbox
    prev_points = points

    results[cur] = {"bbox": init_bbox, "src": "anchor", "conf": 1.0}
    if on_frame is not None:
        on_frame(cur)

    next_idx = cur + step
    while (step > 0 and next_idx <= last) or (step < 0 and next_idx >= last):
        if direction == "forward":
            frame = read_next_frame(cap)
            if frame is None:
                break
            small_frame, scale_back = small_frame_for_tracking(
                frame, max_dim=DEFAULT_TRACKER_MAX_DIM,
            )
        else:
            small_frame = decode_buffered_small_frame(
                backward_small_buffer.get(next_idx)
                if backward_small_buffer is not None else None
            )
            if small_frame is None:
                break
            scale_back = backward_scale_back
            frame = None
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        anchor_candidates = anchors_by_frame.get(next_idx)
        if isinstance(anchor_candidates, dict):
            anchor_candidates = [anchor_candidates]
        if anchor_candidates:
            anchor_reference = last_bbox or smoothed_bbox
            ref_state = bbox_to_state(anchor_reference)
            if ref_state is not None:
                vx, vy, vw, vh = velocity
                cx, cy, bw, bh = ref_state
                predicted_anchor = (
                    max(0.0, cx + vx - max(8.0, bw + vw) / 2.0),
                    max(0.0, cy + vy - max(8.0, bh + vh) / 2.0),
                    min(float(frame_w), cx + vx + max(8.0, bw + vw) / 2.0),
                    min(float(frame_h), cy + vy + max(8.0, bh + vh) / 2.0),
                )
                if predicted_anchor[2] > predicted_anchor[0] and predicted_anchor[3] > predicted_anchor[1]:
                    anchor_reference = predicted_anchor
            anchor_app = choose_anchor_candidate(anchor_candidates, anchor_reference)
            if anchor_app is None:
                next_idx += step
                continue
            anchor_bbox = anchor_app["bbox"]
            smoothed_bbox = anchor_bbox
            last_bbox = anchor_bbox
            try:
                refreshed = create_initialized_tracker(
                    small_frame, anchor_bbox, scale_back, scale_adaptive=True,
                )
                if refreshed is not None:
                    tracker = refreshed
            except Exception:
                pass
            small_bbox = frame_bbox_to_small_bbox(
                anchor_bbox, scale_back, small_frame.shape[1], small_frame.shape[0],
            )
            prev_small_bbox = small_bbox
            prev_points = (
                seed_track_points_for_kind("face", gray_small, small_bbox)
                if small_bbox is not None else None
            )
            prev_gray_small = gray_small
            results[next_idx] = {"bbox": anchor_bbox, "src": "anchor", "conf": 1.0}
            velocity = (0.0, 0.0, 0.0, 0.0)
            if on_frame is not None:
                on_frame(next_idx)
            next_idx += step
            continue

        tracker_bbox = None
        tracker_ok = False
        if tracker is not None:
            try:
                ok, roi = tracker.update(small_frame)
                if ok and roi is not None:
                    x, y, tw, th = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
                    if tw > 0 and th > 0:
                        tracker_bbox = small_bbox_to_frame_bbox(
                            (x, y, x + tw, y + th), scale_back, frame_w, frame_h,
                        )
                        tracker_ok = tracker_bbox is not None
            except Exception:
                tracker_ok = False

        optical_bbox = None
        optical_points = None
        optical_ok = False
        if prev_small_bbox is not None and prev_points is not None and prev_gray_small is not None:
            optical_small_bbox, optical_points = optical_flow_bbox_update(
                prev_gray_small,
                gray_small,
                prev_points,
                prev_small_bbox,
                small_frame.shape[1],
                small_frame.shape[0],
            )
            if optical_small_bbox is not None:
                optical_bbox = small_bbox_to_frame_bbox(
                    optical_small_bbox, scale_back, frame_w, frame_h,
                )
                optical_ok = optical_bbox is not None

        template_bbox = None
        template_score = 0.0
        template_ok = False
        if prev_small_bbox is not None and prev_gray_small is not None:
            template_small_bbox, template_score = template_match_bbox_update(
                prev_gray_small,
                gray_small,
                prev_small_bbox,
                small_frame.shape[1],
                small_frame.shape[0],
                search_expand=2.0,
            )
            if template_small_bbox is not None:
                template_bbox = small_bbox_to_frame_bbox(
                    template_small_bbox, scale_back, frame_w, frame_h,
                )
                template_ok = template_bbox is not None

        global_motion = (
            estimate_global_frame_motion(prev_gray_small, gray_small)
            if prev_gray_small is not None else None
        )
        global_conf = float((global_motion or {}).get("confidence", 0.0) or 0.0)
        global_bbox = None
        if global_motion is not None and global_conf >= 0.14:
            global_bbox = apply_motion_to_bbox(last_bbox, global_motion, frame_w, frame_h)

        # Constant-velocity prediction as a soft anchor when measurement
        # signals briefly drop.
        last_state = bbox_to_state(last_bbox)
        predicted_state = None
        if last_state is not None:
            cx, cy, w, h = last_state
            vx, vy, vw, vh = velocity
            predicted_state = (cx + vx, cy + vy, max(8.0, w + vw), max(8.0, h + vh))
        predicted_bbox = None
        if predicted_state is not None:
            cx, cy, w, h = predicted_state
            x1 = max(0.0, cx - w / 2.0)
            y1 = max(0.0, cy - h / 2.0)
            x2 = min(float(frame_w), cx + w / 2.0)
            y2 = min(float(frame_h), cy + h / 2.0)
            if x2 > x1 and y2 > y1:
                predicted_bbox = (x1, y1, x2, y2)

        # Tighten the fusion: when CSRT clearly disagrees with LK on
        # bbox area (a strong signal that CSRT has grown onto an
        # adjacent texture rather than the face), shrink its weight.
        # LK keeps tighter scale on faces because it is purely
        # point-based on the face's interior corners.
        tracker_scale_weight = (
            scale_disagreement_penalty(tracker_bbox, optical_bbox or template_bbox)
            if tracker_ok and (optical_ok or template_ok) else 1.0
        )

        fused = weighted_fuse_bboxes(
            [
                (optical_bbox, 3.4 if optical_ok else 0.0, "optical"),
                (template_bbox, 2.15 * max(0.5, template_score) if template_ok else 0.0, "template"),
                (tracker_bbox, 2.2 * tracker_scale_weight if tracker_ok else 0.0, "tracker"),
                (
                    global_bbox,
                    1.1 + global_conf if global_bbox is not None else 0.0,
                    "global",
                ),
                (predicted_bbox, 0.7 if predicted_bbox is not None else 0.0, "velocity"),
            ],
            frame_w,
            frame_h,
        )
        if fused is None:
            fused = predicted_bbox or global_bbox or last_bbox

        if fused is None:
            break

        if last_bbox is not None:
            fused = clamp_bbox_scale_step(fused, last_bbox, frame_w, frame_h)

        face_motion = face_motion_rate(velocity, last_state) if last_state is not None else 0.0
        scale_strength = scale_change_strength(velocity, last_state) if last_state is not None else 0.0
        pos_alpha, size_alpha = adaptive_lock_alpha(face_motion, scale_strength)
        smoothed_bbox = smooth_bbox(
            fused,
            smoothed_bbox or last_bbox,
            pos_alpha,
            frame_w,
            frame_h,
            size_alpha=size_alpha,
        ) or fused

        new_state = bbox_to_state(smoothed_bbox)
        velocity = update_velocity(velocity, last_state, new_state)
        last_bbox = smoothed_bbox

        confidence = 0.0
        src = "predicted"
        if optical_ok and tracker_ok:
            confidence = 0.92
            src = "tracker+optical"
        elif optical_ok:
            confidence = 0.85
            src = "optical"
        elif template_ok and tracker_ok:
            confidence = 0.83
            src = "tracker+template"
        elif template_ok:
            confidence = 0.76
            src = "template"
        elif tracker_ok:
            confidence = 0.78
            src = "tracker"
        elif global_bbox is not None:
            confidence = 0.55
            src = "global"
        else:
            confidence = 0.4
            src = "predicted"

        # Dense identity verification: embeddings confirm the selected
        # person, but visual motion remains the primary position signal.
        # A detector bbox only softly corrects an agreeing motion track;
        # it hard-recovers the lane only when motion is weak or lost.
        verified = False
        head_fallback = False
        if (
            known_face is not None
            and verify_interval_frames > 0
            and (abs(next_idx - cur) % verify_interval_frames == 0)
        ):
            verification = verify_face_at_frame(
                frame, smoothed_bbox, known_face, frame_w, frame_h,
            )
            if verification is not None:
                verified_bbox, similarity = verification
                motion_bbox = smoothed_bbox
                motion_agrees = verification_agrees_with_motion(
                    verified_bbox,
                    motion_bbox,
                )[0]
                motion_has_signal = optical_ok or template_ok or tracker_ok or global_bbox is not None
                motion_is_weak = (not motion_has_signal) or confidence < 0.62 or src in ("predicted", "global")
                should_reset_tracker = False

                if motion_agrees:
                    blend = (
                        0.36
                        if similarity >= IDENTITY_VERIFY_HARD_LOCK_SIMILARITY
                        else MOTION_VERIFY_SOFT_BLEND
                    )
                    corrected = smooth_bbox(
                        verified_bbox,
                        motion_bbox,
                        blend,
                        frame_w,
                        frame_h,
                        size_alpha=min(0.32, blend * 0.75),
                    ) or motion_bbox
                    smoothed_bbox = clamp_bbox_scale_step(
                        corrected,
                        motion_bbox,
                        frame_w,
                        frame_h,
                        max_step=0.045,
                    )
                    last_bbox = smoothed_bbox
                    velocity = update_velocity(
                        velocity,
                        last_state,
                        bbox_to_state(smoothed_bbox),
                        alpha=0.28,
                    )
                    verified = True
                    should_reset_tracker = True
                    confidence = max(
                        confidence,
                        0.93 if similarity >= IDENTITY_VERIFY_HARD_LOCK_SIMILARITY else 0.89,
                    )
                    src = "motion_verified"
                elif motion_is_weak:
                    blend = (
                        MOTION_VERIFY_RECOVERY_BLEND
                        if similarity >= IDENTITY_VERIFY_HARD_LOCK_SIMILARITY
                        else 0.58
                    )
                    smoothed_bbox = smooth_bbox(
                        verified_bbox,
                        motion_bbox,
                        blend,
                        frame_w,
                        frame_h,
                        size_alpha=0.52,
                    ) or verified_bbox
                    last_bbox = smoothed_bbox
                    velocity = update_velocity(
                        velocity,
                        last_state,
                        bbox_to_state(smoothed_bbox),
                        alpha=0.58,
                    )
                    verified = True
                    should_reset_tracker = True
                    confidence = max(confidence, 0.91)
                    src = "verified" if similarity >= IDENTITY_VERIFY_HARD_LOCK_SIMILARITY else "motion_verified"
                else:
                    # Identity is present, but the detector box disagrees
                    # with a stable visual track. Keep the motion track to
                    # avoid detector jitter/teleports.
                    confidence = max(confidence, 0.86)
                    src = f"{src}+identity_guard"

                if should_reset_tracker:
                    try:
                        refreshed = create_initialized_tracker(
                            small_frame, smoothed_bbox, scale_back, scale_adaptive=True,
                        )
                        if refreshed is not None:
                            tracker = refreshed
                    except Exception:
                        pass
            else:
                head_match = None
                if abs(next_idx - cur) % HEAD_FALLBACK_INTERVAL_FRAMES == 0:
                    try:
                        from services.detection import localize_head_in_search_region

                        head_search_bbox = expand_search_bbox(
                            smoothed_bbox,
                            IDENTITY_VERIFY_SEARCH_EXPAND * 1.18,
                            frame_w,
                            frame_h,
                        )
                        head_match = localize_head_in_search_region(
                            frame,
                            search_bbox=head_search_bbox,
                            preferred_bbox=smoothed_bbox,
                            strict=True,
                        ) if head_search_bbox is not None else None
                    except Exception:
                        head_match = None

                if head_match is not None:
                    head_bbox = tuple(float(v) for v in head_match["bbox"])
                    smoothed_bbox = smooth_bbox(
                        head_bbox,
                        smoothed_bbox,
                        0.64,
                        frame_w,
                        frame_h,
                        size_alpha=0.54,
                    ) or head_bbox
                    last_bbox = smoothed_bbox
                    velocity = update_velocity(
                        velocity,
                        last_state,
                        bbox_to_state(smoothed_bbox),
                        alpha=0.65,
                    )
                    try:
                        refreshed = create_initialized_tracker(
                            small_frame, smoothed_bbox, scale_back, scale_adaptive=True,
                        )
                        if refreshed is not None:
                            tracker = refreshed
                    except Exception:
                        pass
                    head_fallback = True
                    confidence = max(confidence, 0.74)
                    src = "head_fallback"

        results[next_idx] = {"bbox": smoothed_bbox, "src": src, "conf": confidence}

        small_bbox = frame_bbox_to_small_bbox(
            smoothed_bbox, scale_back, small_frame.shape[1], small_frame.shape[0],
        )
        prev_small_bbox = small_bbox
        # Reseed LK points whenever verification/head fallback corrected
        # the bbox, so optical flow tracks the current head patch and not
        # an old corner cloud.
        if verified or head_fallback:
            prev_points = (
                seed_track_points_for_kind("face", gray_small, small_bbox)
                if small_bbox is not None else None
            )
        elif optical_points is not None and small_bbox is not None:
            x1s, y1s, x2s, y2s = small_bbox
            kept = [
                pt for pt in optical_points.reshape(-1, 2)
                if x1s <= pt[0] <= x2s and y1s <= pt[1] <= y2s
            ]
            if len(kept) >= 6:
                prev_points = np.array(kept, dtype=np.float32).reshape(-1, 1, 2)
            else:
                prev_points = seed_track_points_for_kind("face", gray_small, small_bbox)
        else:
            prev_points = (
                seed_track_points_for_kind("face", gray_small, small_bbox)
                if small_bbox is not None else None
            )
        prev_gray_small = gray_small

        if on_frame is not None:
            on_frame(next_idx)
        next_idx += step

    return results


def is_pinned_src(src):
    """Frames marked anchor / verified are pinned to a real InsightFace
    detection and must not be moved by the cross-direction averaging or
    the bidirectional smoother. They represent the highest-confidence
    bbox we have for that frame.
    """
    return src in ("anchor", "verified")


def is_scale_reference_src(src):
    """Sources trusted enough to stabilize bbox size between detections."""
    return is_pinned_src(src) or src == "head_fallback"


def fuse_directions(forward, backward, frame_w, frame_h):
    """Average forward and backward bbox estimates per frame, weighted by
    each direction's confidence. Anchor / verified frames keep their
    bbox unchanged.
    """
    fused = {}
    keys = set(forward.keys()) | set(backward.keys())
    for f_idx in keys:
        f = forward.get(f_idx)
        b = backward.get(f_idx)
        if f is None and b is None:
            continue
        if f is None:
            fused[f_idx] = dict(b)
            continue
        if b is None:
            fused[f_idx] = dict(f)
            continue
        if is_pinned_src(f.get("src")) and not is_pinned_src(b.get("src")):
            fused[f_idx] = dict(f)
            continue
        if is_pinned_src(b.get("src")) and not is_pinned_src(f.get("src")):
            fused[f_idx] = dict(b)
            continue
        # If both directions verified the same frame, prefer the one
        # with the higher confidence (closer InsightFace match).
        if is_pinned_src(f.get("src")) and is_pinned_src(b.get("src")):
            fused[f_idx] = dict(f if (f.get("conf") or 0) >= (b.get("conf") or 0) else b)
            continue
        fw = max(0.05, float(f.get("conf") or 0.0))
        bw = max(0.05, float(b.get("conf") or 0.0))
        total = fw + bw
        fx1, fy1, fx2, fy2 = f["bbox"]
        bx1, by1, bx2, by2 = b["bbox"]
        x1 = (fx1 * fw + bx1 * bw) / total
        y1 = (fy1 * fw + by1 * bw) / total
        x2 = (fx2 * fw + bx2 * bw) / total
        y2 = (fy2 * fw + by2 * bw) / total
        x1 = max(0.0, min(float(frame_w), x1))
        y1 = max(0.0, min(float(frame_h), y1))
        x2 = max(0.0, min(float(frame_w), x2))
        y2 = max(0.0, min(float(frame_h), y2))
        if x2 <= x1 or y2 <= y1:
            fused[f_idx] = dict(f)
            continue
        fused[f_idx] = {
            "bbox": (x1, y1, x2, y2),
            "src": f"{f.get('src','?')}|{b.get('src','?')}",
            "conf": min(0.99, max(fw, bw)),
        }
    return fused


def stabilize_scale_between_pins(lane_by_frame, frame_w, frame_h):
    """Replace each non-pinned bbox's (width, height) with a linear
    interpolation between the sizes of the surrounding pinned frames
    (anchors / verified). Position is preserved from the tracker fusion
    so the bbox center still moves fluidly with the face, but size is
    forced to the InsightFace-verified scale between every two
    verifications. Result: zero scale drift from CSRT growth.
    """
    if not lane_by_frame:
        return lane_by_frame

    indices = sorted(lane_by_frame.keys())
    scale_indices = [i for i in indices if is_scale_reference_src(lane_by_frame[i].get("src"))]
    if len(scale_indices) < 2:
        return lane_by_frame

    out = {i: dict(entry) for i, entry in lane_by_frame.items()}
    pin_pos = 0
    for i in indices:
        entry = out[i]
        if is_pinned_src(entry.get("src")):
            continue
        # Find the surrounding pinned frames.
        while pin_pos + 1 < len(scale_indices) and scale_indices[pin_pos + 1] <= i:
            pin_pos += 1
        left_i = scale_indices[pin_pos]
        right_i = scale_indices[min(pin_pos + 1, len(scale_indices) - 1)]
        if left_i > i:
            # i is before the first pinned frame: use the first pin's size.
            target_w, target_h = bbox_size(lane_by_frame[scale_indices[0]]["bbox"])
        elif right_i <= i:
            # i is after the last pinned frame: use the last pin's size.
            target_w, target_h = bbox_size(lane_by_frame[scale_indices[-1]]["bbox"])
        else:
            lw, lh = bbox_size(lane_by_frame[left_i]["bbox"])
            rw, rh = bbox_size(lane_by_frame[right_i]["bbox"])
            span = max(1, right_i - left_i)
            t = (i - left_i) / span
            target_w = lw * (1 - t) + rw * t
            target_h = lh * (1 - t) + rh * t

        cx, cy = bbox_center(entry["bbox"])
        if cx is None or target_w <= 0 or target_h <= 0:
            continue
        x1 = max(0.0, cx - target_w / 2.0)
        y1 = max(0.0, cy - target_h / 2.0)
        x2 = min(float(frame_w), cx + target_w / 2.0)
        y2 = min(float(frame_h), cy + target_h / 2.0)
        if x2 <= x1 or y2 <= y1:
            continue
        entry["bbox"] = (x1, y1, x2, y2)

    return out


def bidirectional_smooth(lane_by_frame, frame_w, frame_h, alpha=0.6):
    """Two-pass EMA smoothing across the contiguous lane to remove the
    last bit of high-frequency jitter while preserving anchor / verified
    bbox positions.
    """
    if not lane_by_frame:
        return lane_by_frame

    indices = sorted(lane_by_frame.keys())
    forward = {}
    prev = None
    for f_idx in indices:
        entry = lane_by_frame[f_idx]
        if is_pinned_src(entry.get("src")) or prev is None:
            forward[f_idx] = dict(entry)
            prev = entry["bbox"]
            continue
        smoothed = smooth_bbox(entry["bbox"], prev, alpha, frame_w, frame_h, size_alpha=alpha * 0.7)
        forward[f_idx] = {**entry, "bbox": smoothed or entry["bbox"]}
        prev = smoothed or entry["bbox"]

    backward = {}
    prev = None
    for f_idx in reversed(indices):
        entry = forward[f_idx]
        if is_pinned_src(entry.get("src")) or prev is None:
            backward[f_idx] = dict(entry)
            prev = entry["bbox"]
            continue
        smoothed = smooth_bbox(entry["bbox"], prev, alpha, frame_w, frame_h, size_alpha=alpha * 0.7)
        backward[f_idx] = {**entry, "bbox": smoothed or entry["bbox"]}
        prev = smoothed or entry["bbox"]

    return backward


def safety_pad_ratio_for_bbox(bbox, src=None):
    w, h = bbox_size(bbox)
    max_side = max(w, h)
    pad = FACE_LOCK_SAFETY_PAD_RATIO
    if max_side < 64:
        pad += FACE_LOCK_FAR_HEAD_PAD_RATIO
    elif max_side < 112:
        pad += FACE_LOCK_FAR_HEAD_PAD_RATIO * 0.55
    if src == "head_fallback":
        pad += FACE_LOCK_HEAD_FALLBACK_PAD_RATIO
    return min(0.14, max(0.0, pad))


def apply_safety_pad(bbox, frame_w, frame_h, pad_ratio=None, src=None):
    expanded = expand_face_redaction_bbox(bbox, frame_w, frame_h)
    if not expanded:
        return None
    if pad_ratio is None:
        pad_ratio = safety_pad_ratio_for_bbox(bbox, src=src)
    x1, y1, x2, y2 = expanded
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    pad_top_extra = bh * (pad_ratio * 0.4)
    out = (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y - pad_top_extra),
        min(float(frame_w), x2 + pad_x),
        min(float(frame_h), y2 + pad_y),
    )
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def build_segment_lane(
    cap,
    segment,
    frame_w,
    frame_h,
    fps,
    on_frame=None,
    known_face=None,
):
    """Build a contiguous bbox lane for a single segment using forward
    and backward CSRT/optical-flow/global-motion fusion meeting in the
    middle, dense InsightFace verification anchors, and a bidirectional
    EMA smoothing pass."""

    anchors = segment.get("anchors") or []
    if not anchors:
        return {}

    anchors_by_frame = {}
    for app in anchors:
        f_idx = appearance_frame_index(app, fps)
        anchors_by_frame.setdefault(f_idx, []).append({
            "bbox": app["bbox"],
            "timestamp": app["timestamp"],
        })

    # Forward pass: walk from the first anchor to end_frame, restarting
    # the tracker at every later anchor.
    first_anchor_frame = min(anchors_by_frame.keys())
    last_anchor_frame = max(anchors_by_frame.keys())
    # Use the full segment window, including the TwelveLabs entity range
    # expansion from build_face_lock_segments. This matters when the selected
    # person turns around or moves far from camera: face anchors become sparse,
    # but the tracker should keep the blur on the same head through the whole
    # on-screen presence instead of stopping a short padding window after the
    # last frontal-face detection.
    seg_start = max(0, int(segment["start_frame"]))
    seg_end = max(seg_start, int(segment["end_frame"]))
    seg_start = max(0, seg_start)

    seed_reference_bbox = normalize_appearance_bbox((known_face or {}).get("bbox"))
    first_seed = choose_seed_anchor(
        anchors_by_frame,
        first_anchor_frame,
        direction=1,
        reference_bbox=seed_reference_bbox,
    )
    last_seed = choose_seed_anchor(
        anchors_by_frame,
        last_anchor_frame,
        direction=-1,
        reference_bbox=seed_reference_bbox,
    )
    if first_seed is None or last_seed is None:
        return {}
    first_bbox = first_seed["bbox"]
    last_bbox = last_seed["bbox"]

    seg_t0 = time.monotonic()
    logger.info(
        "Face-lock segment build: frames [%d, %d] (anchors at [%d, %d], %d anchors)",
        seg_start, seg_end, first_anchor_frame, last_anchor_frame, len(anchors_by_frame),
    )

    # Build forward from the first reliable anchor, then backward from the last
    # one. Fusing both directions keeps coverage before the first frontal-face
    # detection and after the last one without trusting a single long tracker run.
    forward_t0 = time.monotonic()
    forward = track_segment_one_direction(
        cap,
        "forward",
        first_bbox,
        first_anchor_frame,
        seg_end,
        frame_w,
        frame_h,
        anchors_by_frame,
        on_frame=on_frame,
        known_face=known_face,
    )
    forward_dt = time.monotonic() - forward_t0
    logger.info(
        "  forward [%d -> %d]: %d frames in %.2fs (%.1f fps)",
        first_anchor_frame, seg_end, len(forward), forward_dt,
        len(forward) / forward_dt if forward_dt > 0 else 0.0,
    )

    backward_t0 = time.monotonic()
    backward = track_segment_one_direction(
        cap,
        "backward",
        last_bbox,
        seg_start,
        last_anchor_frame,
        frame_w,
        frame_h,
        anchors_by_frame,
        on_frame=on_frame,
        known_face=known_face,
    )
    backward_dt = time.monotonic() - backward_t0
    logger.info(
        "  backward [%d <- %d]: %d frames in %.2fs (%.1f fps)",
        seg_start, last_anchor_frame, len(backward), backward_dt,
        len(backward) / backward_dt if backward_dt > 0 else 0.0,
    )

    fused = fuse_directions(forward, backward, frame_w, frame_h)
    # Scale stability before smoothing: lock bbox sizes to the
    # InsightFace-verified scale between pins so CSRT growth onto
    # surrounding textures cannot make the bbox swell.
    scale_stable = stabilize_scale_between_pins(fused, frame_w, frame_h)
    smoothed = bidirectional_smooth(scale_stable, frame_w, frame_h, alpha=0.5)

    logger.info(
        "  fused & smoothed: %d frames, total segment time %.2fs",
        len(smoothed), time.monotonic() - seg_t0,
    )

    return smoothed


def serialize_lane(lane_by_frame, fps, frame_w, frame_h):
    """Convert the per-frame dict into a compact, deterministic JSON
    array. Bboxes are persisted with the safety pad already applied so
    consumers do not need to re-pad.
    """
    indices = sorted(lane_by_frame.keys())
    out = []
    for f_idx in indices:
        entry = lane_by_frame[f_idx]
        bbox = entry.get("bbox")
        if not bbox:
            continue
        padded = apply_safety_pad(bbox, frame_w, frame_h, src=entry.get("src"))
        if padded is None:
            continue
        x1, y1, x2, y2 = padded
        out.append({
            "f": int(f_idx),
            "t": round(float(f_idx) / max(fps, 1.0), 4),
            "x1": round(float(x1), 3),
            "y1": round(float(y1), 3),
            "x2": round(float(x2), 3),
            "y2": round(float(y2), 3),
            "src": entry.get("src") or "track",
            "conf": round(float(entry.get("conf") or 0.0), 4),
        })
    return out


def build_face_lock_lane(
    job_id,
    person_id,
    *,
    force_rebuild=False,
    progress_callback=None,
):
    """Build (or load from cache) the face-lock lane for a person.

    The lane is fully contiguous within each detected segment. The
    visual tracker fusion produces the bbox for every non-anchor frame;
    anchors snap the lane to the InsightFace appearance for that frame
    only. The result is persisted at
    ``backend/snaps/<job_id>/face_lock_tracks/<person_id>.json``.
    """
    from services.pipeline import get_job, get_enriched_faces

    person_id = str(person_id or "").strip()
    if not person_id:
        raise ValueError("person_id is required")

    cached = None if force_rebuild else get_face_lock_lane(job_id, person_id)
    if cached is not None:
        set_build_state(job_id, person_id, status="ready", progress=1.0, percent=100, message="cached")
        return cached

    job = get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    if job.get("status") != "ready":
        raise ValueError(f"Job {job_id} is not ready (status={job.get('status')})")

    video_path = job.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"Job {job_id} has no local video for face-lock build")

    enriched = get_enriched_faces(job_id) or {}
    unique_faces = enriched.get("unique_faces") or job.get("unique_faces") or []
    selected_face = None
    for face in unique_faces:
        if get_face_identity(face) == person_id:
            selected_face = face
            break
    if selected_face is None:
        raise ValueError(f"person_id {person_id} not found in job {job_id}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for face-lock build: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        appearances = collect_person_appearances(selected_face)
        if not appearances:
            raise ValueError(
                f"person_id {person_id} has no stored InsightFace appearances; "
                "cannot build a face-lock lane without anchors"
            )

        video_id = str(job.get("twelvelabs_video_id") or "").strip()
        entity_ranges = get_entity_search_ranges(selected_face, video_id)
        segments = build_face_lock_segments(
            appearances, entity_ranges, fps, total_frames, duration_sec,
        )
        if not segments:
            raise ValueError(
                f"could not build any face-lock segments for person {person_id}"
            )

        # Count the real forward/backward emits so progress reaches 90%
        # predictably even when anchors sit far inside a widened segment.
        expected_emits = 0
        for seg in segments:
            seg_anchors = seg.get("anchors") or []
            if not seg_anchors:
                continue
            anchor_frames = [appearance_frame_index(a, fps) for a in seg_anchors]
            if not anchor_frames:
                continue
            first_anc = min(anchor_frames)
            last_anc = max(anchor_frames)
            seg_start_local = max(0, int(seg["start_frame"]))
            seg_end_local = max(seg_start_local, int(seg["end_frame"]))
            forward_count = max(0, seg_end_local - first_anc + 1)
            backward_count = max(0, last_anc - seg_start_local + 1)
            expected_emits += forward_count + backward_count
        denom_frames = max(1, expected_emits)
        # Reserve the top 10% of the bar (90 -> 100%) for post-loop
        # work (per-segment fusion+smoothing already happens inline,
        # but the global serialize_lane + disk write run after the
        # segment loop and were previously invisible to the user).
        LOOP_PROGRESS_CAP = 0.90
        processed = {"count": 0}
        last_emit_pct = {"value": -1}

        def emit_loop_progress(frame_idx):
            del frame_idx
            processed["count"] += 1
            raw = processed["count"] / denom_frames
            pct = int(round(min(LOOP_PROGRESS_CAP, raw) * 100))
            if pct == last_emit_pct["value"]:
                return
            last_emit_pct["value"] = pct
            set_build_state(
                job_id, person_id,
                status="running",
                progress=round(min(LOOP_PROGRESS_CAP, raw), 4),
                percent=pct,
                message=f"Building face-lock lane ({pct}%)",
            )
            if progress_callback is not None:
                try:
                    progress_callback({
                        "stage": "building",
                        "progress": min(LOOP_PROGRESS_CAP, raw),
                        "percent": pct,
                    })
                except Exception:
                    pass

        def emit_stage_progress(percent, message):
            """Snap the progress bar to a fixed percent for a discrete
            post-loop stage (fusion / serialize / persist). Avoids the
            bar appearing frozen between the last loop emit and the
            final ``ready`` state."""
            pct = int(max(0, min(100, percent)))
            if pct == last_emit_pct["value"]:
                return
            last_emit_pct["value"] = pct
            set_build_state(
                job_id, person_id,
                status="running" if pct < 100 else "ready",
                progress=round(pct / 100.0, 4),
                percent=pct,
                message=message,
            )
            if progress_callback is not None:
                try:
                    progress_callback({
                        "stage": "finalizing" if pct < 100 else "ready",
                        "progress": pct / 100.0,
                        "percent": pct,
                    })
                except Exception:
                    pass

        set_build_state(
            job_id, person_id,
            status="running", progress=0.02, percent=2, message="Initializing trackers",
        )

        all_lane = {}
        all_segment_ranges = []
        for seg in segments:
            seg_lane = build_segment_lane(
                cap, seg, frame_w, frame_h, fps,
                on_frame=emit_loop_progress,
                known_face=selected_face,
            )
            if seg_lane:
                all_lane.update(seg_lane)
                all_segment_ranges.append({
                    "start_frame": min(seg_lane.keys()),
                    "end_frame": max(seg_lane.keys()),
                })

        if not all_lane:
            raise RuntimeError(
                f"face-lock lane build for person {person_id} produced no frames"
            )

        # Loop is done. Push the bar to the LOOP_PROGRESS_CAP so the
        # user sees a clean handoff into the finalize stages even if
        # the actual loop emits stopped a bit short (e.g. an early
        # break on a segment seek failure).
        emit_stage_progress(int(round(LOOP_PROGRESS_CAP * 100)), "Stabilizing lane")
        lane_array = serialize_lane(all_lane, fps, frame_w, frame_h)
        emit_stage_progress(95, "Serializing lane")
        anchor_count = sum(
            1 for entry in all_lane.values() if entry.get("src") == "anchor"
        )
        lane_doc = {
            "build_version": LANE_BUILD_VERSION,
            "job_id": job_id,
            "person_id": person_id,
            "video": {
                "width": frame_w,
                "height": frame_h,
                "fps": float(fps),
                "total_frames": total_frames,
                "duration_sec": float(duration_sec),
            },
            "anchors": {
                "count": anchor_count,
                "from": "insightface_appearances",
            },
            "twelvelabs": {
                "entity_id": str(selected_face.get("entity_id") or "") or None,
                "ranges": [
                    {"start": r[0], "end": r[1]} for r in entity_ranges
                ],
            },
            "segments": all_segment_ranges,
            "lane": lane_array,
            "safety_pad_ratio": FACE_LOCK_SAFETY_PAD_RATIO,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }

        path = face_lock_lane_path(job_id, person_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        emit_stage_progress(98, "Saving lane to disk")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(lane_doc, f, separators=(",", ":"))
        os.replace(tmp_path, path)

        # The dedicated `built_at`/`frames` metadata isn't carried by
        # ``emit_stage_progress`` so set it on the final ready state
        # explicitly. Setting status="ready" + percent=100 also makes
        # the GET endpoint return the lane on the next poll.
        set_build_state(
            job_id, person_id,
            status="ready", progress=1.0, percent=100, message="ready",
            built_at=lane_doc["built_at"],
            frames=len(lane_array),
        )
        last_emit_pct["value"] = 100
        if progress_callback is not None:
            try:
                progress_callback({
                    "stage": "ready",
                    "progress": 1.0,
                    "percent": 100,
                })
            except Exception:
                pass
        logger.info(
            "Built face-lock lane for job=%s person=%s: %d frames across %d segments (%d anchors)",
            job_id, person_id, len(lane_array), len(all_segment_ranges), anchor_count,
        )
        return lane_doc
    finally:
        cap.release()


def lane_bbox_for_frame(lane_doc, frame_idx):
    """Return the (x1, y1, x2, y2) bbox stored for ``frame_idx`` in the
    persisted lane, or None if the frame is not covered.

    Uses bisect-style lookup over the sorted lane array. Lane arrays are
    contiguous within segments so the lookup is exact for covered frames
    and missing for gaps between segments (where the face is off-screen).
    """
    if not lane_doc:
        return None
    lane = lane_doc.get("lane") or []
    if not lane:
        return None
    target = int(frame_idx)
    # Lane is sorted; binary search for the entry with f == target.
    lo, hi = 0, len(lane) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        f = int(lane[mid].get("f", -1))
        if f == target:
            entry = lane[mid]
            return (entry["x1"], entry["y1"], entry["x2"], entry["y2"])
        if f < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


def lane_bbox_at_time(lane_doc, time_sec, *, max_gap_sec=0.06):
    """Linearly interpolate the lane bbox at ``time_sec``. Returns None
    when the requested time is more than ``max_gap_sec`` from any frame
    in the lane (i.e. the face is off-screen)."""
    if not lane_doc:
        return None
    lane = lane_doc.get("lane") or []
    if not lane:
        return None
    fps = float(lane_doc.get("video", {}).get("fps") or 25.0)
    if fps <= 0:
        fps = 25.0
    target_frame = float(time_sec) * fps
    target_frame_int = int(round(target_frame))

    lo, hi = 0, len(lane) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        f = int(lane[mid].get("f", -1))
        if f == target_frame_int:
            entry = lane[mid]
            return (entry["x1"], entry["y1"], entry["x2"], entry["y2"])
        if f < target_frame_int:
            lo = mid + 1
        else:
            hi = mid - 1

    upper_idx = lo
    lower_idx = hi
    upper = lane[upper_idx] if 0 <= upper_idx < len(lane) else None
    lower = lane[lower_idx] if 0 <= lower_idx < len(lane) else None

    def gap_seconds(entry):
        if entry is None:
            return float("inf")
        return abs(int(entry.get("f", -1)) - target_frame) / fps

    upper_gap = gap_seconds(upper)
    lower_gap = gap_seconds(lower)
    if min(upper_gap, lower_gap) > max_gap_sec:
        return None

    if upper is None:
        return (lower["x1"], lower["y1"], lower["x2"], lower["y2"])
    if lower is None:
        return (upper["x1"], upper["y1"], upper["x2"], upper["y2"])

    f_lower = int(lower.get("f", 0))
    f_upper = int(upper.get("f", 0))
    if f_upper == f_lower:
        return (upper["x1"], upper["y1"], upper["x2"], upper["y2"])
    if f_upper - f_lower > int(max_gap_sec * fps) + 1:
        nearest = lower if lower_gap <= upper_gap else upper
        return (nearest["x1"], nearest["y1"], nearest["x2"], nearest["y2"])
    t = (target_frame - f_lower) / max(1.0, float(f_upper - f_lower))
    t = max(0.0, min(1.0, t))
    return (
        lower["x1"] * (1.0 - t) + upper["x1"] * t,
        lower["y1"] * (1.0 - t) + upper["y1"] * t,
        lower["x2"] * (1.0 - t) + upper["x2"] * t,
        lower["y2"] * (1.0 - t) + upper["y2"] * t,
    )
