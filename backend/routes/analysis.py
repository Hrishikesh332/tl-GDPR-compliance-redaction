import base64
import logging
import os
import tempfile

import cv2
import numpy as np
from flask import Blueprint, request, jsonify

from config import OUTPUT_DIR
from services.pipeline import get_job, get_enriched_faces
from services import twelvelabs_service

logger = logging.getLogger("video_redaction.routes.analysis")

analysis_bp = Blueprint("analysis", __name__)


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
