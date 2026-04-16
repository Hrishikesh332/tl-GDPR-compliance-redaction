import base64
import json
import logging
import os
import tempfile
import threading

import cv2
import numpy as np
from flask import Blueprint, request, jsonify

from config import OUTPUT_DIR
from services.redactor import (
    bbox_corners,
    corners_to_bbox,
    detect_best_face_bbox,
    expand_bbox,
    expand_face_redaction_bbox,
    frame_bbox_to_small_bbox,
    normalize_object_class_name,
    optical_flow_bbox_update,
    seed_tracking_points,
    small_bbox_to_frame_bbox,
    translate_bbox,
)
from services.pipeline import get_job, get_enriched_faces
from services.face_identity import ensure_face_identity, get_face_identity
from services import twelvelabs_service
from utils.video import extract_frame_at_time, extract_frames_at_timestamps, small_frame_for_tracking

logger = logging.getLogger("video_redaction.routes.analysis")

analysis_bp = Blueprint("analysis", __name__)

_live_track_state = {}
_live_track_lock = threading.Lock()

LIVE_TRACK_MAX_GAP_SEC = 0.85
LIVE_TRACK_KEEP_CONFIDENCE = 0.12
LIVE_TRACK_DECAY = 0.94
LIVE_TRACK_BLEND_ALPHA = 0.62
LIVE_TRACK_FRAME_MAX_DIM = 640
LIVE_TRACK_GLOBAL_MAX_CORNERS = 240
LIVE_TRACK_GLOBAL_MIN_POINTS = 8
LIVE_TRACK_GLOBAL_MIN_CONFIDENCE = 0.18
LIVE_TRACK_FACE_SEARCH_EXPAND = 1.75
LIVE_TRACK_FACE_LOST_SEARCH_EXPAND = 2.45
LIVE_TRACK_FACE_SEARCH_MOTION_BONUS = 0.55
LIVE_TRACK_FACE_RELOCK_MAX_AREA_RATIO = 3.4
LIVE_TRACK_FACE_RELOCK_MIN_IOU = 0.05
LIVE_TRACK_FACE_RELOCK_MAX_CENTER_SHIFT = 1.15


def normalize_bbox(bbox, frame_w, frame_h):
    if not bbox or frame_w <= 0 or frame_h <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, float(frame_w)))
    y1 = max(0.0, min(y1, float(frame_h)))
    x2 = max(0.0, min(x2, float(frame_w)))
    y2 = max(0.0, min(y2, float(frame_h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "x": round(x1 / frame_w, 6),
        "y": round(y1 / frame_h, 6),
        "width": round((x2 - x1) / frame_w, 6),
        "height": round((y2 - y1) / frame_h, 6),
    }


def denormalize_bbox(box, frame_w, frame_h):
    if not box or frame_w <= 0 or frame_h <= 0:
        return None
    try:
        x = float(box.get("x", 0.0))
        y = float(box.get("y", 0.0))
        width = float(box.get("width", 0.0))
        height = float(box.get("height", 0.0))
    except (TypeError, ValueError):
        return None

    x1 = max(0, min(int(round(x * frame_w)), frame_w))
    y1 = max(0, min(int(round(y * frame_h)), frame_h))
    x2 = max(0, min(int(round((x + width) * frame_w)), frame_w))
    y2 = max(0, min(int(round((y + height) * frame_h)), frame_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def detection_iou(box_a, box_b):
    ax2 = box_a["x"] + box_a["width"]
    ay2 = box_a["y"] + box_a["height"]
    bx2 = box_b["x"] + box_b["width"]
    by2 = box_b["y"] + box_b["height"]
    ix1 = max(box_a["x"], box_b["x"])
    iy1 = max(box_a["y"], box_b["y"])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_a["width"] * box_a["height"] + box_b["width"] * box_b["height"] - inter
    return inter / union if union > 0 else 0.0


def detection_center_distance(box_a, box_b):
    ax = box_a["x"] + box_a["width"] / 2.0
    ay = box_a["y"] + box_a["height"] / 2.0
    bx = box_b["x"] + box_b["width"] / 2.0
    by = box_b["y"] + box_b["height"] / 2.0
    return float(np.hypot(ax - bx, ay - by))


def raw_bbox_area(bbox):
    if not bbox:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def raw_bbox_iou(box_a, box_b):
    if not box_a or not box_b:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = raw_bbox_area(box_a) + raw_bbox_area(box_b) - inter
    return inter / union if union > 0.0 else 0.0


def raw_bbox_center_distance(box_a, box_b):
    if not box_a or not box_b:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    return float(np.hypot(((ax1 + ax2) * 0.5) - ((bx1 + bx2) * 0.5), ((ay1 + ay2) * 0.5) - ((by1 + by2) * 0.5)))


def should_accept_face_relock(candidate_bbox, reference_bbox, search_bbox):
    if candidate_bbox is None:
        return False

    candidate_area = max(1.0, raw_bbox_area(candidate_bbox))
    search_area = max(1.0, raw_bbox_area(search_bbox)) if search_bbox is not None else None
    if reference_bbox is None:
        if search_area is None:
            return True
        return candidate_area <= search_area * 0.82

    reference_area = max(1.0, raw_bbox_area(reference_bbox))
    area_ratio = max(candidate_area, reference_area) / min(candidate_area, reference_area)
    iou = raw_bbox_iou(candidate_bbox, reference_bbox)
    reference_diag = max(
        1.0,
        float(np.hypot(float(reference_bbox[2]) - float(reference_bbox[0]), float(reference_bbox[3]) - float(reference_bbox[1]))),
    )
    center_shift = raw_bbox_center_distance(candidate_bbox, reference_bbox) / reference_diag

    if area_ratio > LIVE_TRACK_FACE_RELOCK_MAX_AREA_RATIO and iou < LIVE_TRACK_FACE_RELOCK_MIN_IOU:
        return False
    if center_shift > LIVE_TRACK_FACE_RELOCK_MAX_CENTER_SHIFT and iou < LIVE_TRACK_FACE_RELOCK_MIN_IOU:
        return False
    if search_area is not None and candidate_area > search_area * 0.82 and iou < LIVE_TRACK_FACE_RELOCK_MIN_IOU:
        return False
    return True


def live_track_match_score(track_det, detection):
    if track_det["kind"] != detection["kind"]:
        return None

    if track_det["kind"] == "face":
        track_person_id = track_det.get("personId")
        detection_person_id = detection.get("personId")
        if track_person_id and detection_person_id and track_person_id != detection_person_id:
            return None
    elif track_det.get("label") != detection.get("label"):
        return None

    iou = detection_iou(track_det, detection)
    distance = detection_center_distance(track_det, detection)
    motion_strength = max(0.0, float(track_det.get("motionStrength", 0.0) or 0.0))
    max_distance = (0.16 if detection["kind"] == "face" else 0.22) + min(0.24, motion_strength * 3.0)
    if track_det.get("trackingMode") == "global":
        max_distance += 0.05
    if iou < 0.04 and distance > max_distance:
        return None

    track_area = max(1e-6, float(track_det["width"]) * float(track_det["height"]))
    det_area = max(1e-6, float(detection["width"]) * float(detection["height"]))
    size_ratio = max(track_area, det_area) / min(track_area, det_area)
    if size_ratio > 3.0 and iou < 0.08:
        return None

    same_identity_bonus = 0.0
    if track_det["kind"] == "face" and track_det.get("personId") and detection.get("personId"):
        same_identity_bonus = 4.0
    if track_det["kind"] == "object" and track_det.get("objectClass") and detection.get("objectClass"):
        same_identity_bonus = 2.0

    return (
        same_identity_bonus
        + iou * 5.0
        - distance * 1.2
        - (size_ratio - 1.0) * 0.2
        + min(0.28, motion_strength * 0.9)
    )


def blend_detection_boxes(track_det, detection):
    motion_strength = max(0.0, float(track_det.get("motionStrength", 0.0) or 0.0))
    alpha = min(
        0.86,
        LIVE_TRACK_BLEND_ALPHA
        + min(0.18, motion_strength * 1.6)
        + (0.08 if track_det.get("trackingMode") == "global" else 0.0),
    )
    return {
        **detection,
        "x": round(float(track_det["x"]) * (1 - alpha) + float(detection["x"]) * alpha, 6),
        "y": round(float(track_det["y"]) * (1 - alpha) + float(detection["y"]) * alpha, 6),
        "width": round(float(track_det["width"]) * (1 - alpha) + float(detection["width"]) * alpha, 6),
        "height": round(float(track_det["height"]) * (1 - alpha) + float(detection["height"]) * alpha, 6),
        "confidence": round(max(float(track_det.get("confidence", 0.0)), float(detection.get("confidence", 0.0))), 4),
    }


def estimate_global_live_motion(prev_gray, gray):
    if prev_gray is None or gray is None:
        return None

    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=LIVE_TRACK_GLOBAL_MAX_CORNERS,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if points is None or len(points) < LIVE_TRACK_GLOBAL_MIN_POINTS:
        return None

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.02),
    )
    if next_points is None or status is None:
        return None

    good_old = points[status.flatten() == 1]
    good_new = next_points[status.flatten() == 1]
    if len(good_old) < LIVE_TRACK_GLOBAL_MIN_POINTS or len(good_new) < LIVE_TRACK_GLOBAL_MIN_POINTS:
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
    feature_ratio = min(1.0, len(good_old) / LIVE_TRACK_GLOBAL_MAX_CORNERS)
    residual_score = 1.0 - min(1.0, (residual or 0.0) / 12.0)
    confidence = max(0.0, min(1.0, feature_ratio * 0.45 + inlier_ratio * 0.35 + residual_score * 0.2))

    return {
        "matrix": matrix,
        "dx": dx,
        "dy": dy,
        "confidence": confidence,
        "motion_strength": motion_strength,
    }


def apply_live_motion_to_bbox(bbox, motion, frame_w, frame_h):
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


def build_tracked_live_detections(
    job_id,
    time_sec,
    frame,
    frame_w,
    frame_h,
    reset_tracking=False,
    known_faces_by_person_id=None,
    face_tolerance=0.35,
):
    small_frame, scale_back = small_frame_for_tracking(frame, max_dim=LIVE_TRACK_FRAME_MAX_DIM)
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    previous_state = None
    known_faces_by_person_id = known_faces_by_person_id or {}

    with _live_track_lock:
        previous_state = _live_track_state.get(job_id)
        if reset_tracking:
            _live_track_state.pop(job_id, None)
            previous_state = None

    if previous_state is None:
        return [], gray_small, scale_back, 0

    previous_time = float(previous_state.get("time_sec", 0.0))
    time_delta = float(time_sec) - previous_time
    if time_delta <= 0.0 or time_delta > LIVE_TRACK_MAX_GAP_SEC:
        return [], gray_small, scale_back, int(previous_state.get("next_track_id", 0))

    previous_gray = previous_state.get("gray_small")
    previous_tracks = previous_state.get("tracks") or []
    global_motion = estimate_global_live_motion(previous_gray, gray_small) if previous_gray is not None else None
    tracked = []

    for track in previous_tracks:
        previous_small_bbox = track.get("small_bbox")
        previous_points = track.get("points")
        previous_frame_bbox = track.get("frame_bbox")
        if previous_small_bbox is None:
            continue

        tracking_mode = "local"
        motion_strength = 0.0
        updated_small_bbox = None
        updated_points = None
        if previous_points is not None:
            updated_small_bbox, updated_points = optical_flow_bbox_update(
                previous_gray,
                gray_small,
                previous_points,
                previous_small_bbox,
                small_frame.shape[1],
                small_frame.shape[0],
            )
        if updated_small_bbox is None and global_motion and float(global_motion.get("confidence", 0.0)) >= LIVE_TRACK_GLOBAL_MIN_CONFIDENCE:
            updated_small_bbox = apply_live_motion_to_bbox(
                previous_small_bbox,
                global_motion,
                small_frame.shape[1],
                small_frame.shape[0],
            )
            updated_points = seed_tracking_points(gray_small, updated_small_bbox) if updated_small_bbox is not None else None
            tracking_mode = "global"
            motion_strength = float(global_motion.get("motion_strength", 0.0) or 0.0)

        if tracking_mode == "local" and global_motion:
            motion_strength = float(global_motion.get("motion_strength", 0.0) or 0.0)

        updated_bbox = small_bbox_to_frame_bbox(updated_small_bbox, scale_back, frame_w, frame_h) if updated_small_bbox is not None else previous_frame_bbox
        if track["kind"] == "face":
            track_person_id = str(track.get("personId") or "").strip()
            search_anchor = updated_bbox or previous_frame_bbox
            if search_anchor is not None:
                search_factor = LIVE_TRACK_FACE_SEARCH_EXPAND if updated_bbox is not None else LIVE_TRACK_FACE_LOST_SEARCH_EXPAND
                search_factor += min(LIVE_TRACK_FACE_SEARCH_MOTION_BONUS, motion_strength * 5.5)
                search_bbox = expand_bbox(search_anchor, frame_w, frame_h, search_factor)
                known_face = known_faces_by_person_id.get(track_person_id) if track_person_id else None
                if known_face is not None:
                    from services.detection import localize_known_face_in_search_region

                    relocked_face = localize_known_face_in_search_region(
                        frame,
                        known_face=known_face,
                        search_bbox=search_bbox,
                        preferred_bbox=updated_bbox or previous_frame_bbox,
                        tolerance=face_tolerance if face_tolerance is not None else 0.35,
                        allow_geometry_fallback=known_face.get("encoding") is None,
                    )
                    relocked_bbox = tuple(relocked_face["bbox"]) if relocked_face is not None else None
                    if relocked_bbox is None or not should_accept_face_relock(
                        relocked_bbox,
                        updated_bbox or previous_frame_bbox,
                        search_bbox,
                    ):
                        continue
                    updated_bbox = relocked_bbox
                    updated_small_bbox = frame_bbox_to_small_bbox(
                        updated_bbox,
                        scale_back,
                        small_frame.shape[1],
                        small_frame.shape[0],
                    )
                    if updated_small_bbox is not None:
                        updated_points = seed_tracking_points(gray_small, updated_small_bbox)
                else:
                    refined_bbox = detect_best_face_bbox(
                        frame,
                        search_bbox,
                        preferred_bbox=updated_bbox or previous_frame_bbox,
                        allow_supplemental=(tracking_mode != "local" or motion_strength >= 0.025),
                    )
                    if refined_bbox is not None and should_accept_face_relock(refined_bbox, updated_bbox or previous_frame_bbox, search_bbox):
                        updated_bbox = refined_bbox
                        updated_small_bbox = frame_bbox_to_small_bbox(
                            updated_bbox,
                            scale_back,
                            small_frame.shape[1],
                            small_frame.shape[0],
                        )
                        if updated_small_bbox is not None:
                            updated_points = seed_tracking_points(gray_small, updated_small_bbox)
        if updated_bbox is None:
            continue
        if updated_small_bbox is None:
            updated_small_bbox = frame_bbox_to_small_bbox(
                updated_bbox,
                scale_back,
                small_frame.shape[1],
                small_frame.shape[0],
            )
        normalized = normalize_bbox(updated_bbox, frame_w, frame_h)
        if normalized is None:
            continue

        tracked.append({
            "kind": track["kind"],
            "label": track["label"],
            "personId": track.get("personId"),
            "objectClass": track.get("objectClass"),
            "confidence": round(max(LIVE_TRACK_KEEP_CONFIDENCE, float(track.get("confidence", 0.0)) * LIVE_TRACK_DECAY), 4),
            "trackId": track["track_id"],
            "trackingMode": tracking_mode,
            "motionStrength": round(motion_strength, 6),
            **normalized,
        })

    return tracked, gray_small, scale_back, int(previous_state.get("next_track_id", 0))


def merge_live_tracked_detections(
    job_id,
    detections,
    tracked_detections,
    gray_small,
    scale_back,
    frame_w,
    frame_h,
    time_sec,
):
    pairs = []
    used_detections = set()
    used_tracks = set()

    for track_idx, track_det in enumerate(tracked_detections):
        for det_idx, detection in enumerate(detections):
            score = live_track_match_score(track_det, detection)
            if score is None:
                continue
            pairs.append((score, track_idx, det_idx))

    pairs.sort(key=lambda item: item[0], reverse=True)
    merged = []
    next_track_id = 0
    with _live_track_lock:
        previous_state = _live_track_state.get(job_id)
        next_track_id = int((previous_state or {}).get("next_track_id", 0))

    for _, track_idx, det_idx in pairs:
        if track_idx in used_tracks or det_idx in used_detections:
            continue
        used_tracks.add(track_idx)
        used_detections.add(det_idx)
        merged.append({
            **blend_detection_boxes(tracked_detections[track_idx], detections[det_idx]),
            "trackId": tracked_detections[track_idx]["trackId"],
        })

    for det_idx, detection in enumerate(detections):
        if det_idx in used_detections:
            continue
        merged.append({
            **detection,
            "trackId": f"{job_id}:{next_track_id}",
        })
        next_track_id += 1

    for track_idx, track_det in enumerate(tracked_detections):
        if track_idx in used_tracks:
            continue
        if float(track_det.get("confidence", 0.0)) < LIVE_TRACK_KEEP_CONFIDENCE:
            continue
        # Keep strong unmatched tracks (including identified faces) so the
        # live blur remains stable between detector refreshes.
        merged.append(track_det)

    next_tracks = []
    for detection in merged:
        bbox = denormalize_bbox(detection, frame_w, frame_h)
        if bbox is None:
            continue
        small_bbox = frame_bbox_to_small_bbox(bbox, scale_back, gray_small.shape[1], gray_small.shape[0])
        points = seed_tracking_points(gray_small, small_bbox) if small_bbox is not None else None
        next_tracks.append({
            "track_id": detection["trackId"],
            "kind": detection["kind"],
            "label": detection["label"],
            "personId": detection.get("personId"),
            "objectClass": detection.get("objectClass"),
            "confidence": float(detection.get("confidence", 0.0)),
            "frame_bbox": bbox,
            "small_bbox": small_bbox,
            "points": points,
        })

    with _live_track_lock:
        _live_track_state[job_id] = {
            "time_sec": float(time_sec),
            "gray_small": gray_small,
            "tracks": next_tracks,
            "next_track_id": next_track_id,
        }

    return merged


def merge_temporal_detections(detections, requested_time):
    grouped = []
    ordered = sorted(
        detections,
        key=lambda det: (abs(float(det.get("sample_time", requested_time)) - requested_time), -float(det.get("confidence", 0.0))),
    )

    for det in ordered:
        best_idx = -1
        best_score = -1e9
        for idx, group in enumerate(grouped):
            if group["kind"] != det["kind"]:
                continue
            if det["kind"] == "face":
                group_person_id = group.get("personId")
                det_person_id = det.get("personId")
                if group_person_id and det_person_id and group_person_id != det_person_id:
                    continue
            if det["kind"] == "object" and group["label"] != det["label"]:
                continue
            if det["kind"] == "object":
                group_object_class = group.get("objectClass")
                det_object_class = det.get("objectClass")
                if group_object_class and det_object_class and group_object_class != det_object_class:
                    continue
            iou = detection_iou(group, det)
            center_distance = detection_center_distance(group, det)
            max_distance = 0.18 if det["kind"] == "face" else 0.24
            if iou < 0.05 and center_distance > max_distance:
                continue
            score = iou * 4.0 - center_distance
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx < 0:
            sample_time = float(det.get("sample_time", requested_time))
            time_weight = 1.0 / (1.0 + abs(sample_time - requested_time) * 12.0)
            confidence = max(0.05, float(det.get("confidence", 0.0)))
            weight = confidence * time_weight
            grouped.append({
                **det,
                "_sum_weight": weight,
                "_sum_x": det["x"] * weight,
                "_sum_y": det["y"] * weight,
                "_sum_width": det["width"] * weight,
                "_sum_height": det["height"] * weight,
                "_members": 1,
            })
            continue

        group = grouped[best_idx]
        sample_time = float(det.get("sample_time", requested_time))
        time_weight = 1.0 / (1.0 + abs(sample_time - requested_time) * 12.0)
        confidence = max(0.05, float(det.get("confidence", 0.0)))
        weight = confidence * time_weight
        group["_sum_weight"] += weight
        group["_sum_x"] += det["x"] * weight
        group["_sum_y"] += det["y"] * weight
        group["_sum_width"] += det["width"] * weight
        group["_sum_height"] += det["height"] * weight
        group["x"] = group["_sum_x"] / group["_sum_weight"]
        group["y"] = group["_sum_y"] / group["_sum_weight"]
        group["width"] = group["_sum_width"] / group["_sum_weight"]
        group["height"] = group["_sum_height"] / group["_sum_weight"]
        group["confidence"] = max(float(group.get("confidence", 0.0)), float(det.get("confidence", 0.0)))
        group["_members"] += 1

    merged = []
    for idx, group in enumerate(grouped):
        merged.append({
            "id": f"{group['kind']}-{idx}",
            "kind": group["kind"],
            "label": group["label"],
            "personId": group.get("personId"),
            "objectClass": group.get("objectClass"),
            "confidence": round(float(group.get("confidence", 0.0)), 4),
            "x": round(float(group["x"]), 6),
            "y": round(float(group["y"]), 6),
            "width": round(float(group["width"]), 6),
            "height": round(float(group["height"]), 6),
        })
    return merged


def filter_detections_to_selected_targets(detections, person_ids=None, object_class_set=None):
    selected_person_ids = {str(item).strip() for item in (person_ids or []) if str(item).strip()}
    selected_object_classes = {
        normalize_object_class_name(item)
        for item in (object_class_set or set())
        if normalize_object_class_name(item)
    }

    if not selected_person_ids and not selected_object_classes:
        return detections

    filtered = []
    for detection in detections or []:
        kind = str(detection.get("kind") or "").strip().lower()
        if kind == "face":
            if not selected_person_ids:
                continue
            person_id = str(detection.get("personId") or "").strip()
            if person_id and person_id in selected_person_ids:
                filtered.append(detection)
                continue
            # Tracked faces may lack a personId between identity re-lock
            # cycles; keep them when the request already scoped to specific
            # person_ids so the blur overlay stays continuous.
            track_id = detection.get("trackId")
            if track_id and not person_id:
                filtered.append(detection)
            continue
        if kind == "object":
            if not selected_object_classes:
                continue
            object_class = normalize_object_class_name(detection.get("objectClass"))
            if object_class and object_class in selected_object_classes:
                filtered.append(detection)
            continue
    return filtered


@analysis_bp.route("/faces/<job_id>", methods=["GET"])
def get_faces(job_id):
    result = get_enriched_faces(job_id)
    if result is None:
        return jsonify({"error": "job not found"}), 404

    if result["status"] == "failed":
        return jsonify({
            "status": result["status"],
            "error": result.get("error") or "Analysis failed for this job.",
        }), 409

    if result["status"] not in ("ready",):
        return jsonify({
            "status": result["status"],
            "message": "Analysis still in progress",
        }), 202

    faces = result.get("unique_faces", [])
    response_faces = []
    for index, f in enumerate(faces):
        stable_person_id = ensure_face_identity(f, fallback_index=index)
        response_faces.append({
            "person_id": stable_person_id,
            "stable_person_id": stable_person_id,
            "name": f.get("name"),
            "snap_base64": f.get("snap_base64"),
            "description": f.get("description", ""),
            "time_ranges": f.get("time_ranges", []),
            "appearance_count": f.get("appearance_count", 0),
            "appearances": f.get("appearances", []),
            "encoding": f.get("encoding"),
            "bbox": f.get("bbox"),
            "entity_id": f.get("entity_id"),
            "entity_asset_ids": f.get("entity_asset_ids", []),
            "tags": f.get("tags", []),
            "should_anonymize": bool(f.get("should_anonymize", False)),
            "is_official": bool(f.get("is_official", False)),
            "priority_rank": f.get("priority_rank"),
        })

    return jsonify({
        "status": "ready",
        "unique_faces": response_faces,
        "entities": result.get("entities", []),
        "total_face_detections": result.get("total_face_detections", 0),
        "twelvelabs_people": result.get("twelvelabs_people"),
        "video_metadata": result.get("video_metadata"),
    })


@analysis_bp.route("/objects/<job_id>", methods=["GET"])
def get_objects(job_id):
    result = get_enriched_faces(job_id)
    if result is None:
        return jsonify({"error": "job not found"}), 404

    if result["status"] == "failed":
        return jsonify({
            "status": result["status"],
            "error": result.get("error") or "Analysis failed for this job.",
        }), 409

    if result["status"] not in ("ready",):
        return jsonify({
            "status": result["status"],
            "message": "Analysis still in progress",
        }), 202

    objects = result.get("unique_objects", [])
    response_objects = []
    for o in objects:
        response_objects.append({
            "object_id": o.get("object_id"),
            "identification": o.get("identification"),
            "snap_base64": o.get("snap_base64"),
            "appearance_count": o.get("appearance_count", 0),
            "appearances": o.get("appearances", []),
            "bbox": o.get("bbox"),
        })

    return jsonify({
        "status": "ready",
        "unique_objects": response_objects,
        "total_object_detections": result.get("total_object_detections", 0),
        "twelvelabs_objects": result.get("twelvelabs_objects"),
        "video_metadata": result.get("video_metadata"),
    })


@analysis_bp.route("/scene-summary/<job_id>", methods=["GET"])
def scene_summary(job_id):
    result = get_enriched_faces(job_id)
    if result is None:
        return jsonify({"error": "job not found"}), 404

    return jsonify({
        "status": result["status"],
        "scene_summary": result.get("twelvelabs_scene_summary"),
        "video_metadata": result.get("video_metadata"),
    })


@analysis_bp.route("/analyze-custom", methods=["POST"])
def analyze_custom():
    data = request.get_json(force=True)
    video_id = data.get("video_id")
    prompt = data.get("prompt")

    if not video_id or not prompt:
        return jsonify({"error": "video_id and prompt are required"}), 400

    result = twelvelabs_service.analyze_video_custom(video_id, prompt)
    return jsonify(result)


@analysis_bp.route("/live-redaction/detect", methods=["POST"])
def live_redaction_detect():
    data = request.get_json(silent=True) or {}

    job_id = data.get("job_id") or request.form.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    if not job.get("video_path"):
        return jsonify({
            "error": "job has no local source video available for live redaction",
            "status": job.get("status"),
        }), 409

    try:
        time_sec = float(data.get("time_sec", request.form.get("time_sec", 0.0)) or 0.0)
    except (TypeError, ValueError):
        time_sec = 0.0
    reset_tracking = str(data.get("reset_tracking", request.form.get("reset_tracking", "false"))).lower() in ("true", "1", "yes")

    include_faces = str(data.get("include_faces", request.form.get("include_faces", "true"))).lower() not in ("false", "0", "no")
    include_objects = str(data.get("include_objects", request.form.get("include_objects", "true"))).lower() not in ("false", "0", "no")
    forensic_only = str(data.get("forensic_only", request.form.get("forensic_only", "true"))).lower() not in ("false", "0", "no")

    person_ids = data.get("person_ids")
    if person_ids is None:
        raw_person_ids = request.form.get("person_ids", "")
        if raw_person_ids:
            try:
                person_ids = json.loads(raw_person_ids)
            except json.JSONDecodeError:
                person_ids = [item.strip() for item in raw_person_ids.split(",") if item.strip()]
    if not isinstance(person_ids, list):
        person_ids = []
    person_ids = [str(item).strip() for item in person_ids if str(item).strip()]
    if person_ids and job.get("status") not in ("ready",):
        return jsonify({
            "error": "saved face identities are still being prepared for this video",
            "status": job.get("status"),
        }), 409

    object_classes = data.get("object_classes")
    if object_classes is None:
        raw_object_classes = request.form.get("object_classes", "")
        if raw_object_classes:
            try:
                object_classes = json.loads(raw_object_classes)
            except json.JSONDecodeError:
                object_classes = [item.strip() for item in raw_object_classes.split(",") if item.strip()]
    if not isinstance(object_classes, list):
        object_classes = []
    object_class_set = {
        normalize_object_class_name(item)
        for item in object_classes
        if normalize_object_class_name(item)
    }

    try:
        object_confidence = float(data.get("object_confidence", request.form.get("object_confidence", 0.25)) or 0.25)
    except (TypeError, ValueError):
        object_confidence = 0.25

    try:
        face_confidence = float(data.get("face_confidence", request.form.get("face_confidence", 0.28)) or 0.28)
    except (TypeError, ValueError):
        face_confidence = 0.28

    # Dense temporal sampling is useful for open-ended "detect everything"
    # preview requests, but it makes selected saved-face playback too slow
    # because the same identity relock work gets repeated across 5 frames.
    # When the editor is following explicit person ids, stay on the current
    # frame and let the saved appearance anchors plus live tracking provide
    # continuity instead.
    sample_offsets = (-0.12, -0.06, 0.0, 0.06, 0.12)
    if person_ids:
        sample_offsets = (0.0,)
    elif object_class_set:
        sample_offsets = (-0.08, 0.0, 0.08)

    sample_times = []
    for offset in sample_offsets:
        ts = max(0.0, time_sec + offset)
        if all(abs(ts - prev) > 1e-4 for prev in sample_times):
            sample_times.append(ts)

    try:
        frame_info = extract_frame_at_time(job["video_path"], time_sec)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    frame = frame_info["frame"]
    frame_h, frame_w = frame.shape[:2]
    if len(sample_times) == 1 and abs(sample_times[0] - float(frame_info["timestamp"])) <= 1e-4:
        frames_by_time = {round(float(frame_info["timestamp"]), 3): frame}
    else:
        sampled_frames = extract_frames_at_timestamps(job["video_path"], sample_times)
        frames_by_time = {round(float(item["timestamp"]), 3): item["frame"] for item in sampled_frames}

    detections = []
    selected_faces = []
    selected_faces_by_person_id = {}

    if person_ids:
        enriched = get_enriched_faces(job_id) or {}
        selected_faces = [
            face for face in (job.get("unique_faces") or enriched.get("unique_faces") or [])
            if get_face_identity(face) in person_ids
        ]
        selected_faces_by_person_id = {
            person_id: face
            for face in selected_faces
            for person_id in [get_face_identity(face)]
            if person_id
        }

    if include_faces:
        if person_ids:
            from services.detection import localize_known_faces_in_frame

            for sample_time in sample_times:
                sample_frame = frames_by_time.get(round(sample_time, 3), frame)
                for face in localize_known_faces_in_frame(sample_frame, selected_faces, time_sec=sample_time):
                    stable_person_id = get_face_identity(face)
                    expanded_bbox = expand_face_redaction_bbox(face.get("bbox"), frame_w, frame_h)
                    normalized = normalize_bbox(expanded_bbox, frame_w, frame_h)
                    if normalized is None:
                        continue
                    detections.append({
                        "kind": "face",
                        "label": str(face.get("name") or stable_person_id or "Face"),
                        "personId": stable_person_id or None,
                        "confidence": round((float(face.get("match_score", 0.0)) * 0.65) + (float(face.get("det_score", 0.0)) * 0.35), 4),
                        "match_score": round(float(face.get("match_score", 0.0)), 4),
                        "sample_time": sample_time,
                        **normalized,
                    })
        else:
            from services.detection import detect_face_boxes

            for sample_time in sample_times:
                sample_frame = frames_by_time.get(round(sample_time, 3), frame)
                for face in detect_face_boxes(
                    sample_frame,
                    confidence_threshold=face_confidence,
                    include_supplemental=True,
                ):
                    expanded_bbox = expand_face_redaction_bbox(face.get("bbox"), frame_w, frame_h)
                    normalized = normalize_bbox(expanded_bbox, frame_w, frame_h)
                    if normalized is None:
                        continue
                    detections.append({
                        "kind": "face",
                        "label": "Face",
                        "personId": None,
                        "confidence": round(float(face.get("det_score", 0.0)), 4),
                        "sample_time": sample_time,
                        **normalized,
                    })

    object_detection_error = None
    if include_objects:
        from services.detection import detect_objects, get_object_detection_error

        for sample_time in sample_times:
            sample_frame = frames_by_time.get(round(sample_time, 3), frame)
            for obj in detect_objects(
                sample_frame,
                conf_threshold=object_confidence,
                forensic_only=forensic_only,
                strict=False,
            ):
                obj_label = str(obj.get("identification") or "Object")
                obj_class = normalize_object_class_name(obj_label)
                if object_class_set and obj_class not in object_class_set:
                    continue
                normalized = normalize_bbox(obj.get("bbox"), frame_w, frame_h)
                if normalized is None:
                    continue
                detections.append({
                    "kind": "object",
                    "label": obj_label,
                    "objectClass": obj_class or obj_label,
                    "confidence": round(float(obj.get("confidence", 0.0)), 4),
                    "sample_time": sample_time,
                    **normalized,
                })
        object_detection_error = get_object_detection_error()

    detections = merge_temporal_detections(detections, time_sec)
    tracked_detections, gray_small, scale_back, _ = build_tracked_live_detections(
        job_id,
        time_sec,
        frame,
        frame_w,
        frame_h,
        reset_tracking=reset_tracking,
        known_faces_by_person_id=selected_faces_by_person_id,
        face_tolerance=0.35,
    )
    detections = merge_live_tracked_detections(
        job_id,
        detections,
        tracked_detections,
        gray_small,
        scale_back,
        frame_w,
        frame_h,
        time_sec,
    )
    detections = filter_detections_to_selected_targets(
        detections,
        person_ids=person_ids,
        object_class_set=object_class_set,
    )

    return jsonify({
        "status": "ready",
        "job_id": job_id,
        "time_sec": frame_info["timestamp"],
        "detections": detections,
        "object_detection_error": object_detection_error,
        "frame": {
            "width": frame_w,
            "height": frame_h,
        },
    })


@analysis_bp.route("/detect-faces", methods=["POST"])
def detect_faces_endpoint():
    """Accept an uploaded image, run face detection, return detected face crops."""
    if "image" not in request.files:
        return jsonify({"error": "missing image file"}), 400

    image_file = request.files["image"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ext = os.path.splitext(image_file.filename or "img.jpg")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(
        dir=OUTPUT_DIR, suffix=ext, delete=False, prefix="detect_"
    )
    image_file.save(tmp.name)
    tmp.close()

    try:
        img = cv2.imread(tmp.name)
        if img is None:
            return jsonify({"error": "could not read image"}), 400

        from services.detection import detect_faces as _detect
        raw_faces = _detect(img, with_encodings=False)

        faces = []
        h_img, w_img = img.shape[:2]
        for i, face in enumerate(raw_faces):
            bbox = face.get("bbox", face.get("box", {}))
            x = int(bbox.get("x", bbox.get("left", 0)))
            y = int(bbox.get("y", bbox.get("top", 0)))
            w = int(bbox.get("w", bbox.get("width", 0)))
            bh = int(bbox.get("h", bbox.get("height", 0)))

            pad = int(max(w, bh) * 0.2)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + bh + pad)
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            _, buf = cv2.imencode(".png", crop)
            b64 = base64.b64encode(buf).decode("ascii")

            faces.append({
                "index": i,
                "confidence": face.get("confidence", 0.9),
                "bbox": {"x": x, "y": y, "w": w, "h": bh},
                "image_base64": b64,
            })

        return jsonify({"faces": faces})
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
