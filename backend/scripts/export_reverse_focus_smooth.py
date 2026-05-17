import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR, OUTPUT_DIR
from services.face_lock_track import build_face_lock_lane, get_face_lock_lane
from services.pipeline import get_job
from services.redactor import (
    bbox_iou,
    expand_bbox,
    expand_face_redaction_bbox,
    prepare_face_lock_export_bboxes,
    union_bboxes,
)
from utils.downloads import safe_redacted_mp4_filename
from utils.image import apply_redaction, restore_region
from utils.video import export_video_dimensions, finalize_mp4_export


JOB_ID = "37896aa0-e99"
FOCUS_PERSON_ID = "person_36"
OUTPUT_HEIGHT = 480
BLUR_STRENGTH = 180
FACE_CONFIDENCE = 0.14
TRACK_LOST_GRACE = 10
TRACK_POS_ALPHA = 0.24
TRACK_SIZE_ALPHA = 0.14
TRACK_VELOCITY_ALPHA = 0.34


def clamp_bbox(bbox, width, height):
    if not bbox:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    x1 = max(0.0, min(x1, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    x2 = max(x1 + 1.0, min(x2, float(width)))
    y2 = max(y1 + 1.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_state(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h)


def state_bbox(state, width, height):
    cx, cy, w, h = [float(v) for v in state[:4]]
    return clamp_bbox((cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0), width, height)


def center_distance(a, b):
    acx, acy, aw, ah = bbox_state(a)
    bcx, bcy, bw, bh = bbox_state(b)
    return float(np.hypot(acx - bcx, acy - bcy)), max(aw, ah, bw, bh)


def nms_boxes(detections, iou_threshold=0.34):
    kept = []
    for bbox, confidence in sorted(detections, key=lambda item: item[1], reverse=True):
        if any(bbox_iou(bbox, existing[0]) >= iou_threshold for existing in kept):
            continue
        kept.append((bbox, confidence))
    return kept


def load_res10_detector():
    proto = os.path.join(BASE_DIR, "models", "deploy.prototxt")
    model = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
    return cv2.dnn.readNetFromCaffe(proto, model)


def detect_faces(frame, net, width, height):
    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < FACE_CONFIDENCE:
            continue
        x1, y1, x2, y2 = (
            detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        ).astype(int)
        raw = clamp_bbox((x1, y1, x2, y2), width, height)
        if raw is None:
            continue
        rx1, ry1, rx2, ry2 = raw
        if (rx2 - rx1) < 5 or (ry2 - ry1) < 5:
            continue
        expanded = expand_face_redaction_bbox(raw, width, height)
        expanded = expand_bbox(expanded, width, height, 1.08)
        expanded = clamp_bbox(expanded, width, height)
        if expanded is not None:
            boxes.append((expanded, confidence))
    return nms_boxes(boxes)


def focus_bbox_for_frame(focus_by_frame, frame_idx, width, height):
    focus_bbox = None
    for entry in focus_by_frame.get(frame_idx) or []:
        bbox = entry.get("bbox")
        if bbox:
            focus_bbox = union_bboxes(focus_bbox, bbox) if focus_bbox else bbox
    if focus_bbox:
        return clamp_bbox(expand_bbox(focus_bbox, width, height, 1.10), width, height)
    return None


def bbox_hits_focus(bbox, focus_bbox, width, height):
    if not bbox or not focus_bbox:
        return False
    focus_gate = expand_bbox(focus_bbox, width, height, 1.22)
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    fx1, fy1, fx2, fy2 = [float(v) for v in focus_gate[:4]]
    if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
        return True
    return bbox_iou(bbox, focus_gate) >= 0.10


def predicted_bbox(track, width, height):
    cx, cy, w, h = bbox_state(track["bbox"])
    vx, vy = track.get("velocity", (0.0, 0.0))
    return state_bbox((cx + vx, cy + vy, w, h), width, height) or track["bbox"]


def match_detections_to_tracks(detections, tracks, width, height):
    candidates = []
    for det_idx, (det_bbox, _confidence) in enumerate(detections):
        for trk_idx, track in enumerate(tracks):
            pred = predicted_bbox(track, width, height)
            iou = bbox_iou(det_bbox, pred)
            dist, scale = center_distance(det_bbox, pred)
            gate = max(28.0, scale * (2.25 if track.get("lost", 0) else 1.75))
            if iou < 0.04 and dist > gate:
                continue
            size_a = bbox_state(det_bbox)[2:]
            size_b = bbox_state(pred)[2:]
            size_ratio = max(size_a[0], size_b[0]) / max(1.0, min(size_a[0], size_b[0]))
            size_ratio += max(size_a[1], size_b[1]) / max(1.0, min(size_a[1], size_b[1]))
            score = iou + max(0.0, 1.0 - dist / gate) * 0.45 - max(0.0, size_ratio - 2.0) * 0.06
            candidates.append((score, det_idx, trk_idx))

    matches = {}
    used_detections = set()
    used_tracks = set()
    for score, det_idx, trk_idx in sorted(candidates, reverse=True):
        if score <= 0.0 or det_idx in used_detections or trk_idx in used_tracks:
            continue
        matches[det_idx] = trk_idx
        used_detections.add(det_idx)
        used_tracks.add(trk_idx)
    return matches, used_detections, used_tracks


def smooth_update_track(track, detection_bbox, confidence, width, height):
    prev_bbox = track["bbox"]
    pcx, pcy, pw, ph = bbox_state(prev_bbox)
    dcx, dcy, dw, dh = bbox_state(detection_bbox)
    old_vx, old_vy = track.get("velocity", (0.0, 0.0))
    predicted_cx = pcx + old_vx
    predicted_cy = pcy + old_vy
    motion = np.hypot(dcx - pcx, dcy - pcy) / max(12.0, np.hypot(pw, ph))
    pos_alpha = min(0.42, TRACK_POS_ALPHA + max(0.0, motion - 0.06) * 1.2)
    size_alpha = min(0.32, TRACK_SIZE_ALPHA + max(0.0, motion - 0.10) * 0.7)
    cx = predicted_cx * (1.0 - pos_alpha) + dcx * pos_alpha
    cy = predicted_cy * (1.0 - pos_alpha) + dcy * pos_alpha
    w = pw * (1.0 - size_alpha) + dw * size_alpha
    h = ph * (1.0 - size_alpha) + dh * size_alpha
    if abs(cx - pcx) < 0.55:
        cx = pcx
    if abs(cy - pcy) < 0.55:
        cy = pcy
    new_bbox = state_bbox((cx, cy, w, h), width, height) or detection_bbox
    new_vx = old_vx * (1.0 - TRACK_VELOCITY_ALPHA) + (dcx - pcx) * TRACK_VELOCITY_ALPHA
    new_vy = old_vy * (1.0 - TRACK_VELOCITY_ALPHA) + (dcy - pcy) * TRACK_VELOCITY_ALPHA
    track.update(
        {
            "bbox": new_bbox,
            "velocity": (new_vx, new_vy),
            "lost": 0,
            "hits": int(track.get("hits", 0)) + 1,
            "age": int(track.get("age", 0)) + 1,
            "confidence": confidence,
        }
    )


def advance_lost_track(track, width, height):
    track["bbox"] = predicted_bbox(track, width, height)
    vx, vy = track.get("velocity", (0.0, 0.0))
    track["velocity"] = (vx * 0.82, vy * 0.82)
    track["lost"] = int(track.get("lost", 0)) + 1
    track["age"] = int(track.get("age", 0)) + 1


def render():
    job = get_job(JOB_ID)
    if not job:
        raise RuntimeError(f"Missing job {JOB_ID}")

    lane = get_face_lock_lane(JOB_ID, FOCUS_PERSON_ID) or build_face_lock_lane(JOB_ID, FOCUS_PERSON_ID)
    if not lane or not lane.get("lane"):
        raise RuntimeError(f"Missing face-lock lane for {FOCUS_PERSON_ID}")

    input_path = job["video_path"]
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_w, output_h = export_video_dimensions(source_w, source_h, OUTPUT_HEIGHT)
    focus_by_frame, focus_stats = prepare_face_lock_export_bboxes(
        {FOCUS_PERSON_ID: lane},
        output_w,
        output_h,
        fps,
        total,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(OUTPUT_DIR, f"redacted_{JOB_ID}_{run_id}_{OUTPUT_HEIGHT}p.mp4")
    fd, temp_path = tempfile.mkstemp(suffix=".mp4", dir=OUTPUT_DIR)
    os.close(fd)
    writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_w, output_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create writer: {temp_path}")

    net = load_res10_detector()
    print(
        json.dumps(
            {
                "event": "start_smooth_reverse_focus_export",
                "job_id": JOB_ID,
                "source": os.path.basename(input_path),
                "focus_person_id": FOCUS_PERSON_ID,
                "width": output_w,
                "height": output_h,
                "total_frames": total,
                "focus_frames": len(focus_by_frame),
                "output_path": output_path,
            },
            indent=2,
        ),
        flush=True,
    )

    tracks = []
    next_track_id = 1
    frame_idx = 0
    last_pct = -1
    face_boxes_seen = 0
    blurred_track_frames = 0
    focus_detections_skipped = 0
    focus_tracks_skipped = 0
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA)
        original = frame.copy()
        focus_bbox = focus_bbox_for_frame(focus_by_frame, frame_idx, output_w, output_h)
        detections = detect_faces(frame, net, output_w, output_h)
        face_boxes_seen += len(detections)

        kept_detections = []
        for bbox, confidence in detections:
            if bbox_hits_focus(bbox, focus_bbox, output_w, output_h):
                focus_detections_skipped += 1
                continue
            kept_detections.append((bbox, confidence))

        matches, used_detections, used_tracks = match_detections_to_tracks(
            kept_detections,
            tracks,
            output_w,
            output_h,
        )

        for det_idx, trk_idx in matches.items():
            bbox, confidence = kept_detections[det_idx]
            smooth_update_track(tracks[trk_idx], bbox, confidence, output_w, output_h)

        for trk_idx, track in enumerate(tracks):
            if trk_idx not in used_tracks:
                advance_lost_track(track, output_w, output_h)

        for det_idx, (bbox, confidence) in enumerate(kept_detections):
            if det_idx in used_detections:
                continue
            tracks.append(
                {
                    "id": next_track_id,
                    "bbox": bbox,
                    "velocity": (0.0, 0.0),
                    "lost": 0,
                    "hits": 1,
                    "age": 1,
                    "confidence": confidence,
                }
            )
            next_track_id += 1

        active_tracks = []
        for track in tracks:
            if int(track.get("lost", 0)) > TRACK_LOST_GRACE:
                continue
            bbox = track.get("bbox")
            if bbox_hits_focus(bbox, focus_bbox, output_w, output_h):
                focus_tracks_skipped += 1
                active_tracks.append(track)
                continue
            render_bbox = expand_bbox(bbox, output_w, output_h, 1.06)
            apply_redaction(frame, render_bbox, "blur", BLUR_STRENGTH, shape="face")
            blurred_track_frames += 1
            active_tracks.append(track)
        tracks = active_tracks

        if focus_bbox:
            restore_region(frame, original, expand_bbox(focus_bbox, output_w, output_h, 1.08))

        writer.write(frame)
        frame_idx += 1

        if total:
            pct = int(frame_idx * 100 / total)
            if pct != last_pct and (pct % 5 == 0 or frame_idx == total):
                last_pct = pct
                elapsed = max(0.001, time.time() - start_time)
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "percent": pct,
                            "frames_processed": frame_idx,
                            "total_frames": total,
                            "active_tracks": len(tracks),
                            "blurred_track_frames": blurred_track_frames,
                            "focus_detections_skipped": focus_detections_skipped,
                            "fps_render": round(frame_idx / elapsed, 2),
                        }
                    ),
                    flush=True,
                )

    cap.release()
    writer.release()

    print(json.dumps({"event": "reencoding", "temp_path": temp_path, "output_path": output_path}), flush=True)
    metadata = finalize_mp4_export(temp_path, output_path, original_path=input_path)
    download_filename = safe_redacted_mp4_filename(os.path.basename(output_path))
    result = {
        "output_path": output_path,
        "download_url": f"/api/download/{download_filename}",
        "download_filename": download_filename,
        "mime_type": "video/mp4",
        "export_quality": f"{OUTPUT_HEIGHT}p",
        "width": output_w,
        "height": output_h,
        "fps": fps,
        "total_frames": frame_idx,
        "face_boxes_seen": face_boxes_seen,
        "blurred_track_frames": blurred_track_frames,
        "focus_detections_skipped": focus_detections_skipped,
        "focus_tracks_skipped": focus_tracks_skipped,
        "focus_export_stats": focus_stats.get(FOCUS_PERSON_ID),
        "output_size_bytes": metadata.get("size_bytes"),
        "h264_encoded": metadata.get("h264_encoded"),
        "download_ready": True,
    }
    print(json.dumps({"event": "done", "result": result}, indent=2), flush=True)


if __name__ == "__main__":
    render()
