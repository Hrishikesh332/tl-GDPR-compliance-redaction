import json
import logging
import os
import time

import cv2
import numpy as np

from config import BASE_DIR, DEFAULT_BLUR_STRENGTH
from services.redactor import (
    bbox_iou,
    expand_bbox,
    expand_face_redaction_bbox,
    prepare_face_lock_export_bboxes,
    smooth_bbox,
    union_bboxes,
)
from utils.image import apply_redaction, restore_region
from utils.video import export_video_dimensions, finalize_mp4_export

logger = logging.getLogger("video_redaction.yolov8_face_reverse")

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n-face-lindevs.pt")
MODEL_PATH = os.environ.get("YOLO_FACE_MODEL", DEFAULT_MODEL_PATH)
YOLO_IMGSZ = int(os.environ.get("YOLO_FACE_IMGSZ", "640") or 640)
YOLO_CONFIDENCE = float(os.environ.get("YOLO_FACE_CONFIDENCE", "0.035") or 0.035)
YOLO_IOU = float(os.environ.get("YOLO_FACE_IOU", "0.34") or 0.34)
YOLO_MAX_DET = int(os.environ.get("YOLO_FACE_MAX_DET", "220") or 220)
YOLO_BATCH = int(os.environ.get("YOLO_FACE_BATCH", "8") or 8)

TRACK_LOST_GRACE = 20
TRACK_BRIDGE_GAP = 22
TRACK_POS_ALPHA = 0.34
TRACK_SIZE_ALPHA = 0.20
TRACK_VELOCITY_ALPHA = 0.34
RENDER_PAD = 1.26

_MODEL_CACHE = {}


def has_yolov8_face_model():
    return os.path.isfile(MODEL_PATH)


def load_yolov8_face_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"YOLOv8-Face model not found: {MODEL_PATH}")
    cached = _MODEL_CACHE.get(MODEL_PATH)
    if cached is not None:
        return cached
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)
    _MODEL_CACHE[MODEL_PATH] = model
    return model


def yolo_device():
    configured = (os.environ.get("YOLO_FACE_DEVICE") or "").strip()
    if configured:
        return configured
    # CPU is slower, but it has been more predictable for full-length exports.
    return "cpu"


def clamp_bbox(bbox, width, height):
    if bbox is None:
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


def nms_boxes(detections, iou_threshold=0.42):
    kept = []
    for bbox, confidence in sorted(detections, key=lambda item: item[1], reverse=True):
        if any(bbox_iou(bbox, existing[0]) >= iou_threshold for existing in kept):
            continue
        kept.append((bbox, confidence))
    return kept


def yolo_result_boxes(result, width, height):
    boxes = []
    if result.boxes is None or result.boxes.xyxy is None:
        return boxes
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones((len(xyxy),))
    for raw, confidence in zip(xyxy, confs):
        raw_bbox = clamp_bbox(raw, width, height)
        if raw_bbox is None:
            continue
        x1, y1, x2, y2 = raw_bbox
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue
        expanded = expand_face_redaction_bbox(raw_bbox, width, height)
        expanded = expand_bbox(expanded, width, height, 1.10)
        expanded = clamp_bbox(expanded, width, height)
        if expanded is not None:
            boxes.append((expanded, float(confidence)))
    return nms_boxes(boxes)


def focus_bbox_for_frame(focus_by_frame, frame_idx, width, height):
    focus_bbox = None
    for entry in focus_by_frame.get(frame_idx) or []:
        bbox = entry.get("bbox")
        if bbox:
            focus_bbox = union_bboxes(focus_bbox, bbox) if focus_bbox else bbox
    if focus_bbox:
        return clamp_bbox(expand_bbox(focus_bbox, width, height, 1.12), width, height)
    return None


def bbox_hits_focus(bbox, focus_bbox, width, height):
    if not bbox or not focus_bbox:
        return False
    focus_gate = expand_bbox(focus_bbox, width, height, 1.28)
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    fx1, fy1, fx2, fy2 = [float(v) for v in focus_gate[:4]]
    if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
        return True
    return bbox_iou(bbox, focus_gate) >= 0.08


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
            gate = max(26.0, scale * (2.35 if track.get("lost", 0) else 1.70))
            if iou < 0.025 and dist > gate:
                continue
            det_w, det_h = bbox_state(det_bbox)[2:]
            pred_w, pred_h = bbox_state(pred)[2:]
            ratio_penalty = (
                max(det_w, pred_w) / max(1.0, min(det_w, pred_w))
                + max(det_h, pred_h) / max(1.0, min(det_h, pred_h))
            )
            score = iou + max(0.0, 1.0 - dist / gate) * 0.46 - max(0.0, ratio_penalty - 2.0) * 0.05
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
    pos_alpha = min(0.52, TRACK_POS_ALPHA + max(0.0, motion - 0.08) * 1.2)
    size_alpha = min(0.38, TRACK_SIZE_ALPHA + max(0.0, motion - 0.10) * 0.8)
    cx = predicted_cx * (1.0 - pos_alpha) + dcx * pos_alpha
    cy = predicted_cy * (1.0 - pos_alpha) + dcy * pos_alpha
    w = pw * (1.0 - size_alpha) + dw * size_alpha
    h = ph * (1.0 - size_alpha) + dh * size_alpha
    if abs(cx - pcx) < 0.45:
        cx = pcx
    if abs(cy - pcy) < 0.45:
        cy = pcy
    new_bbox = state_bbox((cx, cy, w, h), width, height) or detection_bbox
    new_vx = old_vx * (1.0 - TRACK_VELOCITY_ALPHA) + (dcx - pcx) * TRACK_VELOCITY_ALPHA
    new_vy = old_vy * (1.0 - TRACK_VELOCITY_ALPHA) + (dcy - pcy) * TRACK_VELOCITY_ALPHA
    track["bbox"] = new_bbox
    track["velocity"] = (new_vx, new_vy)
    track["lost"] = 0
    track["hits"] = int(track.get("hits", 0)) + 1
    track["age"] = int(track.get("age", 0)) + 1
    track["confidence"] = confidence
    track.setdefault("history", []).append((track["frame_idx"], new_bbox, False))


def advance_lost_track(track, width, height):
    track["bbox"] = predicted_bbox(track, width, height)
    vx, vy = track.get("velocity", (0.0, 0.0))
    track["velocity"] = (vx * 0.86, vy * 0.86)
    track["lost"] = int(track.get("lost", 0)) + 1
    track["age"] = int(track.get("age", 0)) + 1
    track.setdefault("history", []).append((track["frame_idx"], track["bbox"], True))


def build_tracks(input_path, model, device, output_w, output_h, total, focus_by_frame, on_progress=None):
    cap = cv2.VideoCapture(input_path)
    tracks = []
    completed_tracks = []
    next_track_id = 1
    frame_idx = 0
    face_boxes_seen = 0
    focus_detections_skipped = 0
    start_time = time.time()

    while True:
        frames = []
        frame_indices = []
        for _ in range(YOLO_BATCH):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA))
            frame_indices.append(frame_idx)
            frame_idx += 1
        if not frames:
            break

        results = model.predict(
            frames,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU,
            max_det=YOLO_MAX_DET,
            device=device,
            verbose=False,
            batch=min(YOLO_BATCH, len(frames)),
        )

        for local_idx, result in enumerate(results):
            idx = frame_indices[local_idx]
            detections = yolo_result_boxes(result, output_w, output_h)
            face_boxes_seen += len(detections)
            focus_bbox = focus_bbox_for_frame(focus_by_frame, idx, output_w, output_h)
            kept_detections = []
            for bbox, confidence in detections:
                if bbox_hits_focus(bbox, focus_bbox, output_w, output_h):
                    focus_detections_skipped += 1
                    continue
                kept_detections.append((bbox, confidence))

            for track in tracks:
                track["frame_idx"] = idx

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
                        "frame_idx": idx,
                        "history": [(idx, bbox, False)],
                    }
                )
                next_track_id += 1

            active = []
            for track in tracks:
                if int(track.get("lost", 0)) > TRACK_LOST_GRACE:
                    completed_tracks.append(track)
                else:
                    active.append(track)
            tracks = active

        if on_progress:
            elapsed = max(0.001, time.time() - start_time)
            on_progress(
                frame_idx,
                {
                    "active_tracks": len(tracks),
                    "completed_tracks": len(completed_tracks),
                    "face_boxes_seen": face_boxes_seen,
                    "focus_detections_skipped": focus_detections_skipped,
                    "fps_detect": round(frame_idx / elapsed, 2),
                },
            )

    cap.release()
    completed_tracks.extend(tracks)
    return completed_tracks, face_boxes_seen, focus_detections_skipped


def track_to_frame_boxes(track, width, height):
    history = sorted(track.get("history") or [], key=lambda item: int(item[0]))
    detections = [(int(f), bbox) for f, bbox, filled in history if not filled and bbox]
    if not detections:
        return {}

    dense = {}
    for f, bbox in detections:
        dense[f] = bbox

    for (left_f, left_bbox), (right_f, right_bbox) in zip(detections, detections[1:]):
        gap = right_f - left_f
        if gap <= 1 or gap > TRACK_BRIDGE_GAP:
            continue
        l_state = bbox_state(left_bbox)
        r_state = bbox_state(right_bbox)
        for f in range(left_f + 1, right_f):
            t = (f - left_f) / float(gap)
            interp = tuple((1.0 - t) * l_state[i] + t * r_state[i] for i in range(4))
            dense[f] = state_bbox(interp, width, height) or left_bbox

    prev = None
    smoothed = {}
    for f in sorted(dense):
        bbox = dense[f]
        if prev is not None:
            bbox = smooth_bbox(bbox, prev, 0.46, width, height, size_alpha=0.28) or bbox
        smoothed[f] = bbox
        prev = bbox

    reverse_smoothed = {}
    prev = None
    for f in sorted(smoothed, reverse=True):
        bbox = smoothed[f]
        if prev is not None:
            bbox = smooth_bbox(bbox, prev, 0.50, width, height, size_alpha=0.34) or bbox
        reverse_smoothed[f] = bbox
        prev = bbox

    return reverse_smoothed


def build_render_boxes(tracks, focus_by_frame, width, height):
    boxes_by_frame = {}
    for track in tracks:
        if int(track.get("hits", 0)) < 1:
            continue
        for frame_idx, bbox in track_to_frame_boxes(track, width, height).items():
            focus_bbox = focus_bbox_for_frame(focus_by_frame, frame_idx, width, height)
            if bbox_hits_focus(bbox, focus_bbox, width, height):
                continue
            render_bbox = clamp_bbox(expand_bbox(bbox, width, height, RENDER_PAD), width, height)
            if render_bbox is None:
                continue
            boxes_by_frame.setdefault(frame_idx, []).append(render_bbox)

    merged_by_frame = {}
    for frame_idx, boxes in boxes_by_frame.items():
        merged = []
        for bbox in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
            if any(bbox_iou(bbox, existing) >= 0.64 for existing in merged):
                continue
            merged.append(bbox)
        merged_by_frame[frame_idx] = merged
    return merged_by_frame


def render_video(
    input_path,
    output_path,
    boxes_by_frame,
    focus_by_frame,
    output_w,
    output_h,
    fps,
    *,
    blur_strength=DEFAULT_BLUR_STRENGTH,
    redaction_style="blur",
    on_progress=None,
):
    cap = cv2.VideoCapture(input_path)
    root, ext = os.path.splitext(output_path)
    temp_path = f"{root}_opencv{ext or '.mp4'}"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_w, output_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create writer: {temp_path}")

    frame_idx = 0
    redacted_boxes = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA)
        original = frame.copy()
        for bbox in boxes_by_frame.get(frame_idx, ()):
            apply_redaction(frame, bbox, redaction_style, blur_strength, shape="face")
            redacted_boxes += 1
        focus_bbox = focus_bbox_for_frame(focus_by_frame, frame_idx, output_w, output_h)
        if focus_bbox:
            restore_region(frame, original, expand_bbox(focus_bbox, output_w, output_h, 1.08))
        writer.write(frame)
        frame_idx += 1
        if on_progress:
            elapsed = max(0.001, time.time() - start_time)
            on_progress(
                frame_idx,
                {
                    "redacted_boxes": redacted_boxes,
                    "fps_render": round(frame_idx / elapsed, 2),
                },
            )

    cap.release()
    writer.release()
    if not os.path.isfile(temp_path):
        raise RuntimeError(f"OpenCV writer did not create the rendered MP4: {temp_path}")
    metadata = finalize_mp4_export(temp_path, output_path, original_path=input_path)
    return frame_idx, redacted_boxes, metadata


def cache_render_boxes(output_path, boxes_by_frame, width, height, fps, focus_person_ids):
    boxes_cache_path = output_path.replace(".mp4", "_boxes.json")
    with open(boxes_cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(
            {
                "focus_person_ids": focus_person_ids,
                "width": width,
                "height": height,
                "fps": fps,
                "boxes_by_frame": {
                    str(frame_idx): [[round(float(v), 3) for v in bbox] for bbox in boxes]
                    for frame_idx, boxes in boxes_by_frame.items()
                },
            },
            cache_file,
        )
    return boxes_cache_path


def emit_progress(progress_callback, last_emit, stage, progress, frames_processed, total_frames, message=None, **extra):
    if not progress_callback:
        return
    progress = max(0.0, min(1.0, float(progress or 0.0)))
    percent = int(round(progress * 100))
    if last_emit.get("stage") == stage and last_emit.get("percent") == percent:
        return
    last_emit["stage"] = stage
    last_emit["percent"] = percent
    update = {
        "stage": stage,
        "progress": progress,
        "percent": percent,
        "frames_processed": int(frames_processed or 0),
        "total_frames": int(total_frames or 0),
        "message": message,
    }
    update.update(extra)
    progress_callback(update)


def redact_video_yolov8_face_reverse_focus(
    input_path,
    output_path,
    preserve_face_lock_tracks,
    *,
    output_height=720,
    blur_strength=DEFAULT_BLUR_STRENGTH,
    redaction_style="blur",
    progress_callback=None,
):
    if not preserve_face_lock_tracks:
        raise ValueError("YOLOv8 reverse-focus export requires at least one face-lock lane to preserve")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    output_w, output_h = export_video_dimensions(source_w, source_h, output_height)
    focus_by_frame, focus_stats = prepare_face_lock_export_bboxes(
        preserve_face_lock_tracks,
        output_w,
        output_h,
        fps,
        total,
    )
    if not focus_by_frame:
        raise ValueError("YOLOv8 reverse-focus export could not build preserve boxes from the face-lock lane")

    last_emit = {}
    focus_person_ids = sorted(str(pid) for pid in preserve_face_lock_tracks.keys())
    emit_progress(
        progress_callback,
        last_emit,
        "preparing",
        0.02,
        0,
        total,
        "Preparing YOLOv8-Face reverse focus export",
    )

    model = load_yolov8_face_model()
    device = yolo_device()
    logger.info(
        "YOLOv8-Face reverse focus export: model=%s device=%s size=%sx%s frames=%s focus=%s",
        os.path.basename(MODEL_PATH),
        device,
        output_w,
        output_h,
        total,
        focus_person_ids,
    )

    def detect_progress(frames_processed, _stats):
        span = 0.50
        frame_progress = (frames_processed / total) if total else 0.0
        emit_progress(
            progress_callback,
            last_emit,
            "detecting_faces",
            0.05 + span * frame_progress,
            frames_processed,
            total,
            "Detecting faces with YOLOv8-Face",
        )

    tracks, face_boxes_seen, focus_detections_skipped = build_tracks(
        input_path,
        model,
        device,
        output_w,
        output_h,
        total,
        focus_by_frame,
        on_progress=detect_progress,
    )

    boxes_by_frame = build_render_boxes(tracks, focus_by_frame, output_w, output_h)
    boxes_cache_path = None
    if os.environ.get("YOLO_FACE_CACHE_BOXES", "").strip().lower() in {"1", "true", "yes", "on"}:
        boxes_cache_path = cache_render_boxes(output_path, boxes_by_frame, output_w, output_h, fps, focus_person_ids)

    emit_progress(
        progress_callback,
        last_emit,
        "rendering",
        0.57,
        0,
        total,
        "Rendering redacted faces",
    )

    def render_progress(frames_processed, _stats):
        span = 0.34
        frame_progress = (frames_processed / total) if total else 0.0
        emit_progress(
            progress_callback,
            last_emit,
            "rendering",
            0.57 + span * frame_progress,
            frames_processed,
            total,
            "Rendering redacted faces",
        )

    frame_count, redacted_boxes, metadata = render_video(
        input_path,
        output_path,
        boxes_by_frame,
        focus_by_frame,
        output_w,
        output_h,
        fps,
        blur_strength=max(DEFAULT_BLUR_STRENGTH, int(blur_strength or DEFAULT_BLUR_STRENGTH)),
        redaction_style=redaction_style if redaction_style in {"blur", "black"} else "blur",
        on_progress=render_progress,
    )

    emit_progress(
        progress_callback,
        last_emit,
        "completed",
        1.0,
        frame_count,
        total,
        "YOLOv8-Face reverse focus export complete",
    )

    return {
        "output_path": output_path,
        "width": output_w,
        "height": output_h,
        "source_width": source_w,
        "source_height": source_h,
        "fps": fps,
        "total_frames": frame_count,
        "detection_frames_processed": total,
        "detection_frames_skipped": 0,
        "output_size_bytes": metadata.get("size_bytes"),
        "h264_encoded": metadata.get("h264_encoded"),
        "download_ready": True,
        "reverse_face_redaction_engine": "yolov8_face",
        "preserve_face_lock_export_stats": focus_stats,
        "yolov8_face_export_stats": {
            "model": os.path.basename(MODEL_PATH),
            "device": device,
            "imgsz": YOLO_IMGSZ,
            "confidence": YOLO_CONFIDENCE,
            "iou": YOLO_IOU,
            "batch": YOLO_BATCH,
            "tracks": len(tracks),
            "render_frames_with_boxes": len(boxes_by_frame),
            "face_boxes_seen": face_boxes_seen,
            "focus_detections_skipped": focus_detections_skipped,
            "redacted_boxes": redacted_boxes,
            "boxes_cache_path": boxes_cache_path,
            "focus_frames": len(focus_by_frame),
            "focus_person_ids": focus_person_ids,
        },
    }
