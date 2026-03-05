import logging
import os
import sys
import tempfile

import cv2
import numpy as np

from config import TRACKER_MAX_DIM, DEFAULT_BLUR_STRENGTH, DEFAULT_DETECT_EVERY_N
from utils.image import apply_blur
from utils.video import small_frame_for_tracking, reencode_mp4_to_h264

logger = logging.getLogger("video_redaction.redactor")


def _create_tracker(scale_adaptive=False):
    """Create a tracker. If scale_adaptive=True, use CSRT for better resize with motion (when available)."""
    if scale_adaptive:
        try:
            return cv2.legacy.TrackerCSRT_create()
        except (AttributeError, cv2.error):
            try:
                return cv2.TrackerCSRT_create()
            except (AttributeError, cv2.error):
                pass
    try:
        return cv2.legacy.TrackerKCF_create()
    except AttributeError:
        return cv2.TrackerKCF_create()


def _tracker_roi_to_frame_bbox(x, y, tw, th, scale_back, frame_w, frame_h, min_side=4):
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


def _match_objects(frame_bgr, target_classes, conf_threshold=0.25):
    if not target_classes:
        return []
    from services.detection import _get_obj_model
    model = _get_obj_model()
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


def _normalized_region_to_bbox(region, w, h):
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


def redact_video(
    input_path,
    output_path,
    face_encodings=None,
    object_classes=None,
    face_tolerance=None,
    obj_conf=0.25,
    blur_strength=DEFAULT_BLUR_STRENGTH,
    detect_every_n=DEFAULT_DETECT_EVERY_N,
    detect_every_seconds=None,
    temporal_ranges=None,
    custom_regions=None,
):
    face_encodings = face_encodings or []
    object_classes = object_classes or set()
    temporal_ranges = temporal_ranges or []
    custom_regions = custom_regions or []

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if detect_every_seconds is not None and detect_every_seconds > 0:
        detect_every_n = max(1, int(round(fps * detect_every_seconds)))

    detect_every_n = max(1, min(detect_every_n, 10))

    logger.info("Redact: %dx%d, %.1f fps, ~%d frames, detect_every=%d, targets: %d faces, %d obj_classes, %d custom (tracked)",
                w, h, fps, total, detect_every_n, len(face_encodings), len(object_classes), len(custom_regions))

    tmp_fd, temp_path = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(output_path) or ".")
    os.close(tmp_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

    scale_back = max(w, h) / TRACKER_MAX_DIM if max(w, h) > TRACKER_MAX_DIM else 1.0
    trackers = []
    frame_idx = 0
    detection_frames_processed = 0

    last_detected_boxes = []

    # Motion tracking for manual (drawn) regions: tracker per region, init on frame 0, update every frame.
    # Each entry: (tracker or None for static fallback, effect, static_bbox, last_bbox from tracker or None).
    custom_trackers = []  # list of tracker or None
    custom_effects = []
    custom_static_bboxes = []  # initial bbox (x1,y1,x2,y2) for static fallback when tracker missing/failed
    custom_last_bboxes = []    # last successful tracker bbox; used when update() fails so blur keeps following
    custom_fail_count = []     # consecutive update() failures per region; used to re-init after several failures
    CUSTOM_TRACKER_REINIT_INTERVAL = 20   # re-init tracker every N frames so it keeps following (avoids freeze)
    CUSTOM_TRACKER_REINIT_AFTER_FAILS = 5  # re-init when update() fails this many times in a row

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = frame_idx / fps
        in_temporal_range = True
        if temporal_ranges:
            in_temporal_range = any(
                r["start"] <= current_sec <= r["end"]
                for r in temporal_ranges
            )

        # Init custom-region trackers on first frame (manual blur with motion tracking)
        if frame_idx == 0 and custom_regions:
            small, _ = small_frame_for_tracking(frame, TRACKER_MAX_DIM)
            for reg in custom_regions:
                bbox = _normalized_region_to_bbox(reg, w, h)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                apply_blur(frame, (x1, y1, x2, y2), blur_strength)
                effect = reg.get("effect") or "blur"
                tracker = None
                try:
                    tr = _create_tracker(scale_adaptive=True)
                    sx = int(x1 / scale_back)
                    sy = int(y1 / scale_back)
                    sw = max(1, int((x2 - x1) / scale_back))
                    sh = max(1, int((y2 - y1) / scale_back))
                    sx = max(0, min(sx, small.shape[1] - 1))
                    sy = max(0, min(sy, small.shape[0] - 1))
                    sw = max(1, min(sw, small.shape[1] - sx))
                    sh = max(1, min(sh, small.shape[0] - sy))
                    tr.init(small, (sx, sy, sw, sh))
                    tracker = tr
                except cv2.error as e:
                    logger.warning("Custom region tracker init failed, using static bbox: %s", e)
                custom_trackers.append(tracker)
                custom_effects.append(effect)
                custom_static_bboxes.append(bbox)
                custom_last_bboxes.append(bbox)
                custom_fail_count.append(0)

        # Apply custom regions every frame: track or fall back to static/last bbox
        if custom_static_bboxes and frame_idx > 0:
            new_trackers = []
            new_effects = []
            new_static = []
            new_last = []
            new_fail_count = []
            small, scale_back_actual = small_frame_for_tracking(frame, TRACKER_MAX_DIM)
            periodic_reinit = (frame_idx % CUSTOM_TRACKER_REINIT_INTERVAL == 0)
            for idx, (tr, eff, static_bbox, last_bbox) in enumerate(zip(
                custom_trackers, custom_effects, custom_static_bboxes, custom_last_bboxes
            )):
                fail_count = custom_fail_count[idx] if idx < len(custom_fail_count) else 0
                if tr is None:
                    apply_blur(frame, static_bbox, blur_strength)
                    new_trackers.append(None)
                    new_effects.append(eff)
                    new_static.append(static_bbox)
                    new_last.append(last_bbox)
                    new_fail_count.append(0)
                    continue
                use_bbox = last_bbox if last_bbox else static_bbox
                # Periodic re-init so tracker keeps following (avoids stuck bbox)
                if periodic_reinit and use_bbox:
                    try:
                        x1, y1, x2, y2 = use_bbox
                        sx = max(0, min(int(x1 / scale_back_actual), small.shape[1] - 1))
                        sy = max(0, min(int(y1 / scale_back_actual), small.shape[0] - 1))
                        sw = max(1, min(int((x2 - x1) / scale_back_actual), small.shape[1] - sx))
                        sh = max(1, min(int((y2 - y1) / scale_back_actual), small.shape[0] - sy))
                        tr.init(small, (sx, sy, sw, sh))
                        fail_count = 0
                    except cv2.error:
                        pass
                ok, roi = tr.update(small)
                if ok:
                    x, y, tw, th = [int(v) for v in roi]
                    bbox = _tracker_roi_to_frame_bbox(x, y, tw, th, scale_back_actual, w, h)
                    if bbox:
                        apply_blur(frame, bbox, blur_strength)
                        new_last.append(bbox)
                    else:
                        new_last.append(last_bbox)
                    new_trackers.append(tr)
                    new_effects.append(eff)
                    new_static.append(static_bbox)
                    new_fail_count.append(0)
                else:
                    # Tracker lost: show last bbox; re-init after several failures so it can recover
                    if use_bbox:
                        apply_blur(frame, use_bbox, blur_strength)
                    if fail_count >= CUSTOM_TRACKER_REINIT_AFTER_FAILS and use_bbox:
                        try:
                            x1, y1, x2, y2 = use_bbox
                            sx = max(0, min(int(x1 / scale_back_actual), small.shape[1] - 1))
                            sy = max(0, min(int(y1 / scale_back_actual), small.shape[0] - 1))
                            sw = max(1, min(int((x2 - x1) / scale_back_actual), small.shape[1] - sx))
                            sh = max(1, min(int((y2 - y1) / scale_back_actual), small.shape[0] - sy))
                            tr.init(small, (sx, sy, sw, sh))
                            fail_count = 0
                        except cv2.error:
                            pass
                    new_trackers.append(tr)
                    new_effects.append(eff)
                    new_static.append(static_bbox)
                    new_last.append(last_bbox)
                    new_fail_count.append(fail_count + 1)
            custom_trackers = new_trackers
            custom_effects = new_effects
            custom_static_bboxes = new_static
            custom_last_bboxes = new_last
            custom_fail_count = new_fail_count

        run_detection = (
            frame_idx % detect_every_n == 0
            and in_temporal_range
            and (face_encodings or object_classes)
        )

        if run_detection:
            from services.detection import match_faces_in_frame
            face_boxes = match_faces_in_frame(frame, face_encodings) if face_encodings else []
            obj_boxes = _match_objects(frame, object_classes, conf_threshold=obj_conf) if object_classes else []
            all_boxes = face_boxes + obj_boxes
            detection_frames_processed += 1

            trackers = []
            small, _ = small_frame_for_tracking(frame, TRACKER_MAX_DIM)
            for box in all_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                apply_blur(frame, (x1, y1, x2, y2), blur_strength)

                tracker = _create_tracker(scale_adaptive=True)
                sx = int(x1 / scale_back)
                sy = int(y1 / scale_back)
                sw = max(1, int((x2 - x1) / scale_back))
                sh = max(1, int((y2 - y1) / scale_back))
                sx = max(0, sx)
                sy = max(0, sy)
                tracker.init(small, (sx, sy, sw, sh))
                trackers.append(tracker)

            last_detected_boxes = all_boxes
        else:
            if trackers:
                small, _ = small_frame_for_tracking(frame, TRACKER_MAX_DIM)
                new_trackers = []
                for tracker in trackers:
                    ok, roi = tracker.update(small)
                    if ok:
                        x, y, tw, th = [int(v) for v in roi]
                        bbox = _tracker_roi_to_frame_bbox(x, y, tw, th, scale_back, w, h)
                        if bbox:
                            apply_blur(frame, bbox, blur_strength)
                        new_trackers.append(tracker)
                trackers = new_trackers
            elif last_detected_boxes and in_temporal_range:
                for box in last_detected_boxes:
                    apply_blur(frame, box, blur_strength)

        writer.write(frame)
        frame_idx += 1

        if total > 0 and frame_idx % max(1, total // 10) == 0:
            pct = int(100 * frame_idx / total)
            logger.info("Redact progress: %d%% (%d/%d)", pct, frame_idx, total)

    cap.release()
    writer.release()

    reencode_mp4_to_h264(temp_path, output_path)
    try:
        os.remove(temp_path)
    except OSError:
        pass

    logger.info("Redaction complete: %d frames, %d detection passes -> %s",
                frame_idx, detection_frames_processed, output_path)
    return {
        "output_path": output_path,
        "total_frames": frame_idx,
        "fps": fps,
        "width": w,
        "height": h,
        "detection_frames_processed": detection_frames_processed,
        "detection_frames_skipped": 0,
    }
