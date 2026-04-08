import json
import logging

from flask import Blueprint, request, jsonify

from services.detection import ObjectDetectionUnavailable
from services.pipeline import run_redaction, preview_redaction_tracks, get_job, get_enriched_faces

logger = logging.getLogger("video_redaction.routes.redaction")

redaction_bp = Blueprint("redaction", __name__)


def parse_custom_regions(data):
    custom_regions = data.get("custom_regions")
    if custom_regions is None and request.form.get("custom_regions"):
        try:
            custom_regions = json.loads(request.form.get("custom_regions"))
        except (TypeError, json.JSONDecodeError):
            custom_regions = []
    if not isinstance(custom_regions, list):
        custom_regions = []
    return custom_regions


@redaction_bp.route("/redact", methods=["POST"])
def redact():
    data = request.get_json(silent=True) or {}

    job_id = data.get("job_id") or request.form.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    if job["status"] != "ready":
        return jsonify({
            "error": "job is not ready for redaction",
            "status": job["status"],
        }), 409

    person_ids = data.get("person_ids")
    if person_ids is None:
        raw_pids = request.form.get("person_ids", "")
        if raw_pids:
            try:
                person_ids = json.loads(raw_pids)
            except json.JSONDecodeError:
                person_ids = [s.strip() for s in raw_pids.split(",") if s.strip()]

    face_encodings = data.get("face_encodings")
    if face_encodings is None:
        raw = request.form.get("face_encodings", "")
        if raw:
            try:
                face_encodings = json.loads(raw)
            except json.JSONDecodeError:
                face_encodings = []

    face_targets = []
    if person_ids:
        enriched = get_enriched_faces(job_id) or {}
        unique_faces = job.get("unique_faces") or enriched.get("unique_faces", [])
        if not face_encodings:
            face_encodings = []
        matched_names = []
        for face in unique_faces:
            pid = face.get("person_id", "")
            if pid in person_ids:
                face_targets.append(face)
                enc = face.get("encoding")
                if enc:
                    face_encodings.append(enc)
                    matched_names.append(pid)
        logger.info("Resolved person_ids %s → %d encodings (%s)",
                    person_ids, len(face_encodings), matched_names)

    object_classes = data.get("object_classes")
    if object_classes is None:
        raw_obj = request.form.get("object_classes", "")
        if raw_obj:
            object_classes = [s.strip() for s in raw_obj.split(",") if s.strip()]

    try:
        blur_strength = int(data.get("blur_strength", request.form.get("blur_strength", 51)))
    except (TypeError, ValueError):
        blur_strength = 51

    redaction_style = str(data.get("redaction_style", request.form.get("redaction_style", "blur")) or "blur").strip().lower()
    if redaction_style not in {"blur", "black"}:
        redaction_style = "blur"

    try:
        detect_every_n = int(data.get("detect_every_n", request.form.get("detect_every_n", 3)))
    except (TypeError, ValueError):
        detect_every_n = 3

    detect_every_seconds = data.get("detect_every_seconds")
    if detect_every_seconds is None:
        raw_sec = request.form.get("detect_every_seconds", "")
        if raw_sec:
            try:
                detect_every_seconds = float(raw_sec)
            except (TypeError, ValueError):
                detect_every_seconds = None

    use_temporal = data.get("use_temporal_optimization", True)

    entity_ids = data.get("entity_ids")
    if entity_ids is None:
        raw_entities = request.form.get("entity_ids", "")
        if raw_entities:
            try:
                entity_ids = json.loads(raw_entities)
            except json.JSONDecodeError:
                entity_ids = [s.strip() for s in raw_entities.split(",") if s.strip()]

    custom_regions = parse_custom_regions(data)

    total_targets = max(len(face_targets), len(face_encodings) if face_encodings else 0) + (len(object_classes) if object_classes else 0) + len(custom_regions)
    if total_targets == 0 and not entity_ids:
        return jsonify({"error": "No targets selected. Provide person_ids, face_encodings, object_classes, entity_ids, or custom_regions (drawn regions with motion tracking)."}), 400

    logger.info("Redacting job %s: %d face targets / %d face encodings (person_ids=%s), %s object classes, %s entity_ids, %d custom_regions",
                job_id,
                len(face_targets) if face_targets else 0,
                len(face_encodings) if face_encodings else 0,
                person_ids or "none",
                object_classes or "none",
                len(entity_ids) if entity_ids else 0,
                len(custom_regions))

    try:
        result = run_redaction(
            job_id=job_id,
            face_encodings=face_encodings,
            face_targets=face_targets,
            object_classes=object_classes,
            entity_ids=entity_ids,
            custom_regions=custom_regions,
            blur_strength=blur_strength,
            redaction_style=redaction_style,
            detect_every_n=detect_every_n,
            detect_every_seconds=detect_every_seconds,
            use_temporal_optimization=use_temporal,
        )
    except ObjectDetectionUnavailable as e:
        return jsonify({"error": str(e)}), 503

    return jsonify({
        "output_path": result["output_path"],
        "download_url": result["download_url"],
        "total_frames": result["total_frames"],
        "fps": result["fps"],
        "detection_frames_processed": result.get("detection_frames_processed", 0),
        "detection_frames_skipped": result.get("detection_frames_skipped", 0),
        "entity_ids_used": result.get("entity_ids_used", []),
        "temporal_ranges_from_entity_search": result.get("temporal_ranges_from_entity_search", 0),
        "person_ids_used": person_ids or [],
    })


@redaction_bp.route("/redact/preview-track", methods=["POST"])
def preview_track():
    data = request.get_json(silent=True) or {}

    job_id = data.get("job_id") or request.form.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    if job["status"] != "ready":
        return jsonify({
            "error": "job is not ready for preview tracking",
            "status": job["status"],
        }), 409

    custom_regions = parse_custom_regions(data)
    if not custom_regions:
        return jsonify({"error": "Provide at least one custom region for preview tracking."}), 400

    try:
        preview_fps = float(data.get("preview_fps", request.form.get("preview_fps", 8)))
    except (TypeError, ValueError):
        preview_fps = 8.0

    logger.info("Preview tracking job %s: %d custom_regions at %.2f fps", job_id, len(custom_regions), preview_fps)
    try:
        result = preview_redaction_tracks(job_id=job_id, custom_regions=custom_regions, preview_fps=preview_fps)
        return jsonify(result)
    except Exception as e:
        logger.error("Preview tracking failed for job %s: %s", job_id, str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500
