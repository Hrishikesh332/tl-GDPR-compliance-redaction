import logging
import os
import time

import cv2

from config import DEFAULT_BLUR_STRENGTH
from services.redactor import (
    bbox_iou,
    expand_bbox,
    expand_face_redaction_bbox,
    prepare_face_lock_export_bboxes,
    smooth_bbox,
)
from services.yolov8_face_reverse import (
    MODEL_PATH,
    YOLO_BATCH,
    YOLO_CONFIDENCE,
    YOLO_IMGSZ,
    YOLO_IOU,
    center_distance,
    clamp_bbox,
    emit_progress,
    has_yolov8_face_model,
    load_yolov8_face_model,
    nms_boxes,
    yolo_device,
)
from utils.image import apply_redaction
from utils.video import export_video_dimensions, finalize_mp4_export

logger = logging.getLogger("video_redaction.yolov8_face_target")

TARGET_GATE_EXPAND = 1.34
TARGET_RENDER_PAD = 1.035
LANE_RENDER_PAD = 1.12
FACE_PRESENT_ONLY = os.environ.get("YOLO_FACE_TARGET_FACE_PRESENT_ONLY", "1").strip().lower() not in {"0", "false", "no", "off"}
PREVIOUS_TRACK_MAX_GAP = 8 if FACE_PRESENT_ONLY else 30
HISTORY_BRIDGE_MAX_FRAMES = 4 if FACE_PRESENT_ONLY else 18
HISTORY_BRIDGE_PAD_PER_FRAME = 0.012 if FACE_PRESENT_ONLY else 0.022
HISTORY_BRIDGE_PAD_CAP = 0.055 if FACE_PRESENT_ONLY else 0.20
AMBIGUOUS_SCORE_MARGIN = 0.16


def bbox_dimensions(bbox):
    if not bbox:
        return 0.0, 0.0
    return (
        max(1.0, float(bbox[2]) - float(bbox[0])),
        max(1.0, float(bbox[3]) - float(bbox[1])),
    )


def bbox_diag(bbox):
    w, h = bbox_dimensions(bbox)
    return max(1.0, (w * w + h * h) ** 0.5)


def source_is_weak(entry):
    source = str((entry or {}).get("src") or "").lower()
    if (entry or {}).get("filled"):
        return True
    if source in {"anchor", "motion_verified"}:
        return False
    return any(
        token in source
        for token in (
            "bridge",
            "interpolated",
            "held",
            "motion_only",
            "global",
            "head_fallback",
            "head_walk",
        )
    )


def lane_render_bbox(bbox, width, height):
    expanded = expand_face_redaction_bbox(bbox, width, height)
    expanded = expand_bbox(expanded, width, height, LANE_RENDER_PAD)
    return clamp_bbox(expanded, width, height)


def yolo_render_bbox(bbox, width, height):
    expanded = expand_face_redaction_bbox(bbox, width, height)
    expanded = expand_bbox(expanded, width, height, TARGET_RENDER_PAD)
    return clamp_bbox(expanded, width, height)


def target_yolo_face_boxes(result, width, height):
    boxes = []
    if result.boxes is None or result.boxes.xyxy is None:
        return boxes
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else [1.0] * len(xyxy)
    for raw, confidence in zip(xyxy, confs):
        bbox = clamp_bbox(raw, width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue
        boxes.append((bbox, float(confidence)))
    return nms_boxes(boxes, iou_threshold=0.38)


def bbox_center_inside(bbox, gate_bbox):
    if not bbox or not gate_bbox:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    gx1, gy1, gx2, gy2 = [float(v) for v in gate_bbox[:4]]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return gx1 <= cx <= gx2 and gy1 <= cy <= gy2


def choose_target_detection(yolo_boxes, lane_bbox, used_indices, width, height, previous_bbox=None, entry=None):
    if not yolo_boxes or not lane_bbox:
        return None, None, "no_detections"

    gate_bbox = clamp_bbox(expand_bbox(lane_bbox, width, height, TARGET_GATE_EXPAND), width, height)
    if not gate_bbox:
        return None, None, "invalid_gate"

    best = None
    second_best = None
    lane_diag = bbox_diag(lane_bbox)
    lane_dist_gate = max(10.0, min(42.0, lane_diag * 0.58))
    weak_without_history = source_is_weak(entry) and previous_bbox is None
    prev_dist_gate = None
    if previous_bbox is not None:
        prev_dist_gate = max(12.0, min(52.0, bbox_diag(previous_bbox) * 0.88))

    for idx, (bbox, confidence) in enumerate(yolo_boxes):
        if idx in used_indices:
            continue
        gate_overlap = bbox_iou(bbox, gate_bbox)
        lane_overlap = bbox_iou(bbox, lane_bbox)
        if gate_overlap <= 0.0 and not bbox_center_inside(bbox, gate_bbox):
            continue
        dist, scale = center_distance(bbox, lane_bbox)
        del scale
        center_tight = bbox_center_inside(bbox, clamp_bbox(expand_bbox(lane_bbox, width, height, 1.16), width, height))
        if lane_overlap < 0.08 and dist > lane_dist_gate and not center_tight:
            continue
        if weak_without_history and (lane_overlap < 0.16 or dist > max(8.0, lane_diag * 0.38)):
            continue
        prev_overlap = 0.0
        if previous_bbox is not None:
            prev_overlap = bbox_iou(bbox, previous_bbox)
            prev_dist, _prev_scale = center_distance(bbox, previous_bbox)
            if prev_overlap < 0.08 and prev_dist > prev_dist_gate:
                continue
        score = (
            lane_overlap * 2.6
            + gate_overlap * 0.55
            + max(0.0, 1.0 - dist / lane_dist_gate) * 0.82
            + prev_overlap * 1.55
            + float(confidence or 0.0) * 0.16
        )
        if best is None or score > best[0]:
            second_best = best
            best = (score, idx, bbox)
        elif second_best is None or score > second_best[0]:
            second_best = (score, idx, bbox)

    if best is None:
        return None, None, "no_safe_match"
    score_margin = 0.22 if weak_without_history else AMBIGUOUS_SCORE_MARGIN
    if second_best is not None and (best[0] - second_best[0]) < score_margin:
        return None, None, "ambiguous_match"
    min_score = 0.74 if weak_without_history else 0.48
    if best[0] < min_score:
        return None, None, "weak_match"
    return best[2], best[1], "matched"


def has_nearby_unaccepted_face(yolo_boxes, lane_bbox, width, height):
    if not yolo_boxes or not lane_bbox:
        return False
    lane_diag = bbox_diag(lane_bbox)
    conflict_gate = max(16.0, min(58.0, lane_diag * 1.05))
    broad_gate = clamp_bbox(expand_bbox(lane_bbox, width, height, 1.48), width, height)
    for bbox, _confidence in yolo_boxes:
        overlap = bbox_iou(bbox, broad_gate) if broad_gate is not None else 0.0
        dist, _scale = center_distance(bbox, lane_bbox)
        if overlap >= 0.05 or dist <= conflict_gate:
            return True
    return False


def lane_fallback_is_safe(entry, lane_bbox, previous_bbox, yolo_boxes, width, height):
    if FACE_PRESENT_ONLY:
        return False
    if lane_bbox is None:
        return False
    if previous_bbox is None:
        # If YOLO does not see a competing face near the selected-person lane,
        # using the lane is safer than revealing the selected face.
        return not has_nearby_unaccepted_face(yolo_boxes, lane_bbox, width, height)
    prev_overlap = bbox_iou(lane_bbox, previous_bbox)
    prev_dist, _scale = center_distance(lane_bbox, previous_bbox)
    prev_gate = max(14.0, min(54.0, bbox_diag(previous_bbox) * 0.92))
    if prev_overlap < 0.10 and prev_dist > prev_gate:
        return False
    if has_nearby_unaccepted_face(yolo_boxes, lane_bbox, width, height) and source_is_weak(entry):
        return False
    return True


def grouped_entries_by_person(entries):
    by_person = {}
    for entry in entries or []:
        pid = str(entry.get("person_id") or "").strip()
        bbox = entry.get("bbox")
        if not pid or not bbox:
            continue
        by_person.setdefault(pid, []).append(entry)
    return by_person


def previous_person_bbox(person_id, person_states, frame_idx):
    state = person_states.get(person_id) or {}
    previous = state.get("bbox")
    previous_frame = state.get("frame_idx")
    if previous is None or previous_frame is None:
        return None
    if int(frame_idx) - int(previous_frame) > PREVIOUS_TRACK_MAX_GAP:
        return None
    if state.get("source") == "bridge" and int(state.get("bridge_count", 0) or 0) >= HISTORY_BRIDGE_MAX_FRAMES:
        return None
    return previous


def bridge_bbox_from_history(person_id, person_states, frame_idx, lane_bbox, lane_render, width, height):
    state = person_states.get(person_id) or {}
    previous = state.get("bbox")
    previous_frame = state.get("frame_idx")
    if previous is None or previous_frame is None:
        return None
    age = int(frame_idx) - int(previous_frame)
    if age <= 0 or age > PREVIOUS_TRACK_MAX_GAP:
        return None
    bridge_count = int(state.get("bridge_count", 0) or 0)
    if bridge_count >= HISTORY_BRIDGE_MAX_FRAMES:
        return None
    if FACE_PRESENT_ONLY and state.get("source") not in {"yolo", "bridge"}:
        return None

    reference_lane = lane_render or lane_bbox
    if reference_lane is not None:
        overlap = bbox_iou(reference_lane, previous)
        dist, _scale = center_distance(reference_lane, previous)
        dist_gate = max(12.0, min(36.0, bbox_diag(previous) * (0.72 if FACE_PRESENT_ONLY else 1.35)))
        min_overlap = 0.10 if FACE_PRESENT_ONLY else 0.035
        if overlap < min_overlap and dist > dist_gate:
            return None

    pad = 1.0 + min(HISTORY_BRIDGE_PAD_CAP, HISTORY_BRIDGE_PAD_PER_FRAME * max(1, bridge_count + age))
    bridged = expand_bbox(previous, width, height, pad)
    if reference_lane is not None:
        lane_overlap = bbox_iou(reference_lane, previous)
        if lane_overlap >= 0.06:
            bridged = smooth_bbox(reference_lane, bridged, 0.20, width, height, size_alpha=0.16) or bridged
    return clamp_bbox(bridged, width, height)


def smooth_person_bbox(person_id, candidate_bbox, person_states, width, height, source, frame_idx):
    previous = previous_person_bbox(person_id, person_states, frame_idx)
    if previous is not None:
        if source == "yolo":
            alpha, size_alpha = 0.58, 0.34
        elif source == "bridge":
            alpha, size_alpha = 0.24, 0.18
        else:
            alpha, size_alpha = 0.48, 0.28
        candidate_bbox = smooth_bbox(candidate_bbox, previous, alpha, width, height, size_alpha=size_alpha) or candidate_bbox
    previous_state = person_states.get(person_id) or {}
    person_states[person_id] = {
        "bbox": candidate_bbox,
        "frame_idx": int(frame_idx),
        "source": source,
        "bridge_count": int(previous_state.get("bridge_count", 0) or 0) + 1 if source == "bridge" else 0,
    }
    return candidate_bbox


def redact_video_yolov8_face_targets(
    input_path,
    output_path,
    face_lock_tracks,
    *,
    output_height=720,
    blur_strength=DEFAULT_BLUR_STRENGTH,
    redaction_style="blur",
    progress_callback=None,
    frame_limit=None,
    start_frame=0,
):
    if not face_lock_tracks:
        raise ValueError("YOLOv8-Face target export requires at least one selected face-lock lane")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(start_frame or 0))
    if source_total > 0:
        start_frame = min(start_frame, max(0, source_total - 1))
    render_total = max(0, source_total - start_frame)
    if frame_limit:
        render_total = min(render_total, int(frame_limit))

    output_w, output_h = export_video_dimensions(source_w, source_h, output_height)
    target_by_frame, face_lock_stats = prepare_face_lock_export_bboxes(
        face_lock_tracks,
        output_w,
        output_h,
        fps,
        source_total,
    )
    if not target_by_frame:
        cap.release()
        raise ValueError("YOLOv8-Face target export could not build target boxes from the selected face-lock lane")

    root, ext = os.path.splitext(output_path)
    temp_path = f"{root}_opencv{ext or '.mp4'}"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_w, output_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create writer: {temp_path}")

    last_emit = {}
    model = load_yolov8_face_model()
    device = yolo_device()
    selected_person_ids = sorted(str(pid) for pid in face_lock_tracks.keys())
    emit_progress(
        progress_callback,
        last_emit,
        "detecting_faces",
        0.04,
        0,
        render_total,
        "Detecting selected faces with YOLOv8-Face",
    )
    logger.info(
        "YOLOv8-Face target export: model=%s device=%s size=%sx%s frames=%s people=%s",
        os.path.basename(MODEL_PATH),
        device,
        output_w,
        output_h,
        render_total,
        selected_person_ids,
    )

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    processed_frames = 0
    yolo_faces_seen = 0
    target_entries_seen = 0
    yolo_matched_boxes = 0
    lane_fallback_boxes = 0
    history_bridge_boxes = 0
    suppressed_ambiguous_matches = 0
    suppressed_lane_fallbacks = 0
    suppressed_temporal_jumps = 0
    redacted_boxes = 0
    frames_with_target = 0
    person_states = {}
    start_time = time.time()
    style = redaction_style if redaction_style in {"blur", "black"} else "blur"
    strength = max(DEFAULT_BLUR_STRENGTH, int(blur_strength or DEFAULT_BLUR_STRENGTH))

    try:
        while True:
            frames = []
            frame_indices = []
            for _ in range(YOLO_BATCH):
                if frame_limit and processed_frames >= int(frame_limit):
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA))
                frame_indices.append(frame_idx)
                frame_idx += 1
                processed_frames += 1
            if not frames:
                break

            results = model.predict(
                frames,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                max_det=220,
                device=device,
                verbose=False,
                batch=min(YOLO_BATCH, len(frames)),
            )

            for local_idx, result in enumerate(results):
                idx = frame_indices[local_idx]
                frame = frames[local_idx]
                entries_by_person = grouped_entries_by_person(target_by_frame.get(idx, ()))
                yolo_boxes = target_yolo_face_boxes(result, output_w, output_h)
                yolo_faces_seen += len(yolo_boxes)
                used_indices = set()
                if entries_by_person:
                    frames_with_target += 1

                for person_id, entries in entries_by_person.items():
                    for entry in entries:
                        target_entries_seen += 1
                        lane_match_bbox = clamp_bbox(entry.get("bbox"), output_w, output_h)
                        lane_bbox = lane_render_bbox(entry.get("bbox"), output_w, output_h)
                        if lane_bbox is None or lane_match_bbox is None:
                            continue
                        previous_bbox = previous_person_bbox(person_id, person_states, idx)
                        yolo_bbox, yolo_idx, match_reason = choose_target_detection(
                            yolo_boxes,
                            lane_match_bbox,
                            used_indices,
                            output_w,
                            output_h,
                            previous_bbox=previous_bbox,
                            entry=entry,
                        )
                        if yolo_bbox is not None:
                            used_indices.add(yolo_idx)
                            render_bbox = yolo_render_bbox(yolo_bbox, output_w, output_h)
                            source = "yolo"
                            yolo_matched_boxes += 1
                        else:
                            if match_reason == "ambiguous_match":
                                suppressed_ambiguous_matches += 1
                            safe_fallback = lane_fallback_is_safe(
                                entry,
                                lane_match_bbox,
                                previous_bbox,
                                yolo_boxes,
                                output_w,
                                output_h,
                            )
                            if not safe_fallback:
                                bridge_bbox = bridge_bbox_from_history(
                                    person_id,
                                    person_states,
                                    idx,
                                    lane_match_bbox,
                                    lane_bbox,
                                    output_w,
                                    output_h,
                                )
                                if bridge_bbox is not None:
                                    render_bbox = bridge_bbox
                                    source = "bridge"
                                    history_bridge_boxes += 1
                                elif previous_bbox is not None:
                                    prev_overlap = bbox_iou(lane_bbox, previous_bbox)
                                    prev_dist, _scale = center_distance(lane_bbox, previous_bbox)
                                    prev_gate = max(14.0, min(54.0, bbox_diag(previous_bbox) * 0.92))
                                    if prev_overlap < 0.10 and prev_dist > prev_gate:
                                        suppressed_temporal_jumps += 1
                                    else:
                                        suppressed_lane_fallbacks += 1
                                    continue
                                else:
                                    suppressed_lane_fallbacks += 1
                                    continue
                            else:
                                render_bbox = lane_bbox
                                source = "lane"
                                lane_fallback_boxes += 1
                        if render_bbox is None:
                            continue
                        render_bbox = smooth_person_bbox(
                            person_id,
                            render_bbox,
                            person_states,
                            output_w,
                            output_h,
                            source,
                            idx,
                        )
                        apply_redaction(frame, render_bbox, style, strength, shape="face")
                        redacted_boxes += 1

                writer.write(frame)

            if render_total:
                elapsed = max(0.001, time.time() - start_time)
                progress = min(0.91, 0.04 + 0.87 * (min(processed_frames, render_total) / max(render_total, 1)))
                emit_progress(
                    progress_callback,
                    last_emit,
                    "rendering",
                    progress,
                    processed_frames,
                    render_total,
                    "Rendering selected-face blur with YOLOv8-Face",
                    fps_render=round(processed_frames / elapsed, 2),
                )
    finally:
        cap.release()
        writer.release()

    if not os.path.isfile(temp_path):
        raise RuntimeError(f"OpenCV writer did not create the rendered MP4: {temp_path}")

    emit_progress(
        progress_callback,
        last_emit,
        "reencoding",
        0.94,
        processed_frames,
        render_total,
        "Re-encoding output",
    )
    metadata = finalize_mp4_export(temp_path, output_path, original_path=input_path)
    emit_progress(
        progress_callback,
        last_emit,
        "completed",
        1.0,
        processed_frames,
        render_total,
        "YOLOv8-Face selected-face export complete",
    )

    return {
        "output_path": output_path,
        "width": output_w,
        "height": output_h,
        "source_width": source_w,
        "source_height": source_h,
        "fps": fps,
        "total_frames": processed_frames,
        "detection_frames_processed": processed_frames,
        "detection_frames_skipped": 0,
        "output_size_bytes": metadata.get("size_bytes"),
        "h264_encoded": metadata.get("h264_encoded"),
        "download_ready": True,
        "normal_face_redaction_engine": "yolov8_face",
        "face_lock_export_stats": face_lock_stats,
        "yolov8_face_target_export_stats": {
            "model": os.path.basename(MODEL_PATH),
            "device": device,
            "imgsz": YOLO_IMGSZ,
            "confidence": YOLO_CONFIDENCE,
            "iou": YOLO_IOU,
            "batch": YOLO_BATCH,
            "face_present_only": FACE_PRESENT_ONLY,
            "history_bridge_max_frames": HISTORY_BRIDGE_MAX_FRAMES,
            "selected_person_ids": selected_person_ids,
            "source_start_frame": start_frame,
            "source_total_frames": source_total,
            "frames_with_target": frames_with_target,
            "target_entries_seen": target_entries_seen,
            "yolo_faces_seen": yolo_faces_seen,
            "yolo_matched_boxes": yolo_matched_boxes,
            "lane_fallback_boxes": lane_fallback_boxes,
            "history_bridge_boxes": history_bridge_boxes,
            "suppressed_ambiguous_matches": suppressed_ambiguous_matches,
            "suppressed_lane_fallbacks": suppressed_lane_fallbacks,
            "suppressed_temporal_jumps": suppressed_temporal_jumps,
            "redacted_boxes": redacted_boxes,
        },
    }


__all__ = [
    "has_yolov8_face_model",
    "redact_video_yolov8_face_targets",
]
