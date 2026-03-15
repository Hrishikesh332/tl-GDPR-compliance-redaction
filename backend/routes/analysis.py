import base64
import json
import logging
import os
import tempfile

import cv2
import numpy as np
from flask import Blueprint, request, jsonify

from config import OUTPUT_DIR
from services.pipeline import get_job, get_enriched_faces
from services import twelvelabs_service
from utils.video import extract_frame_at_time, extract_frames_at_timestamps

logger = logging.getLogger("video_redaction.routes.analysis")

analysis_bp = Blueprint("analysis", __name__)


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


@analysis_bp.route("/faces/<job_id>", methods=["GET"])
def get_faces(job_id):
    result = get_enriched_faces(job_id)
    if result is None:
        return jsonify({"error": "job not found"}), 404

    if result["status"] not in ("ready",):
        return jsonify({
            "status": result["status"],
            "message": "Analysis still in progress",
        }), 202

    faces = result.get("unique_faces", [])
    response_faces = []
    for f in faces:
        response_faces.append({
            "person_id": f.get("person_id"),
            "snap_base64": f.get("snap_base64"),
            "description": f.get("description", ""),
            "time_ranges": f.get("time_ranges", []),
            "appearance_count": f.get("appearance_count", 0),
            "appearances": f.get("appearances", []),
            "encoding": f.get("encoding"),
            "bbox": f.get("bbox"),
            "entity_id": f.get("entity_id"),
            "entity_asset_ids": f.get("entity_asset_ids", []),
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

    if job["status"] not in ("ready",):
        return jsonify({
            "error": "job is not ready for live redaction detection",
            "status": job["status"],
        }), 409

    try:
        time_sec = float(data.get("time_sec", request.form.get("time_sec", 0.0)) or 0.0)
    except (TypeError, ValueError):
        time_sec = 0.0

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
    object_class_set = {str(item).strip() for item in object_classes if str(item).strip()}

    try:
        object_confidence = float(data.get("object_confidence", request.form.get("object_confidence", 0.25)) or 0.25)
    except (TypeError, ValueError):
        object_confidence = 0.25

    try:
        face_confidence = float(data.get("face_confidence", request.form.get("face_confidence", 0.28)) or 0.28)
    except (TypeError, ValueError):
        face_confidence = 0.28

    sample_offsets = (-0.12, -0.06, 0.0, 0.06, 0.12) if person_ids else (-0.10, 0.0, 0.10)
    sample_times = []
    for offset in sample_offsets:
        ts = max(0.0, time_sec + offset)
        if all(abs(ts - prev) > 1e-4 for prev in sample_times):
            sample_times.append(ts)

    try:
        frame_info = extract_frame_at_time(job["video_path"], time_sec)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    sampled_frames = extract_frames_at_timestamps(job["video_path"], sample_times)
    frames_by_time = {round(float(item["timestamp"]), 3): item["frame"] for item in sampled_frames}

    frame = frame_info["frame"]
    frame_h, frame_w = frame.shape[:2]
    detections = []

    if include_faces:
        if person_ids:
            from services.detection import identify_faces_in_frame

            selected_faces = [
                face for face in (job.get("unique_faces") or [])
                if str(face.get("person_id") or "").strip() in person_ids and face.get("encoding") is not None
            ]
            for sample_time in sample_times:
                sample_frame = frames_by_time.get(round(sample_time, 3), frame)
                for face in identify_faces_in_frame(sample_frame, selected_faces):
                    normalized = normalize_bbox(face.get("bbox"), frame_w, frame_h)
                    if normalized is None:
                        continue
                    detections.append({
                        "kind": "face",
                        "label": str(face.get("person_id") or "Face"),
                        "personId": str(face.get("person_id") or "").strip() or None,
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
                    normalized = normalize_bbox(face.get("bbox"), frame_w, frame_h)
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
                if object_class_set and obj_label not in object_class_set:
                    continue
                normalized = normalize_bbox(obj.get("bbox"), frame_w, frame_h)
                if normalized is None:
                    continue
                detections.append({
                    "kind": "object",
                    "label": obj_label,
                    "objectClass": obj_label,
                    "confidence": round(float(obj.get("confidence", 0.0)), 4),
                    "sample_time": sample_time,
                    **normalized,
                })
        object_detection_error = get_object_detection_error()

    detections = merge_temporal_detections(detections, time_sec)

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
