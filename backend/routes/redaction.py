import base64
import json
import logging
import threading
import uuid

import cv2
import numpy as np

from flask import Blueprint, request, jsonify

from config import DEFAULT_BLUR_STRENGTH, DEFAULT_DETECT_EVERY_N
from services.detection import ObjectDetectionUnavailable, detect_uploaded_reference_faces
from services.face_identity import ensure_face_identity
from services.pipeline import run_redaction, preview_redaction_tracks, get_job, get_enriched_faces
from utils.video import EXPORT_HEIGHTS, normalize_export_height

logger = logging.getLogger("video_redaction.routes.redaction")

redaction_bp = Blueprint("redaction", __name__)
_redaction_jobs = {}
_redaction_jobs_lock = threading.Lock()


class RedactionRequestError(Exception):
    def __init__(self, message, status_code=400, payload=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}


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


def parse_list_field(data, key, *, split_csv=False):
    value = data.get(key)
    if value is None:
        raw_value = request.form.get(key, "")
        if raw_value:
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = [item.strip() for item in raw_value.split(",")] if split_csv else []
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if split_csv and isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def parse_bool_field(data, key, default=False):
    value = data.get(key)
    if value is None:
        value = request.form.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def parse_output_height(data):
    raw_value = (
        data.get("output_height")
        or data.get("export_height")
        or data.get("export_quality")
        or request.form.get("output_height")
        or request.form.get("export_quality")
    )
    if raw_value in (None, ""):
        return normalize_export_height(None)
    try:
        if isinstance(raw_value, str):
            parsed = int(raw_value.strip().lower().removesuffix("p"))
        else:
            parsed = int(raw_value)
    except (TypeError, ValueError):
        raise RedactionRequestError("export_quality must be one of 480p, 720p, or 1080p", 400)
    if parsed not in EXPORT_HEIGHTS:
        raise RedactionRequestError("export_quality must be one of 480p, 720p, or 1080p", 400)
    return parsed


def custom_region_is_face(region):
    if not isinstance(region, dict):
        return False
    mode = str(region.get("tracking_mode", "")).strip().lower()
    if mode == "face":
        return True
    reason = str(region.get("reason", "")).lower()
    return any(token in reason for token in ("face", "person", "head"))


def _decode_face_image_base64(image_b64):
    """Decode a base64-encoded face image to a BGR numpy array.

    Accepts either the raw base64 payload or a full data URL ("data:...;base64,...").
    Returns ``None`` if the data cannot be decoded into an image.
    """
    if not image_b64 or not isinstance(image_b64, str):
        return None
    payload = image_b64.strip()
    if "," in payload and payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=False)
    except (ValueError, TypeError):
        return None
    if not raw:
        return None
    try:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None or img.size == 0:
        return None
    return img


def parse_face_images(data):
    """Parse and validate the optional ``face_images`` field on a redact request.

    Accepts a list of ``{person_id, label, image_base64}`` entries. Each entry's
    base64 image is run through the InsightFace pipeline so we can attach a
    real encoding for per-frame redaction at export time. Used for entities
    that the user added via "Snap face from video" — those have no
    pre-existing entry in the job's ``unique_faces`` list, so the only way
    to redact them is to compute the encoding now from the image they
    snapped.

    Returns ``(face_targets, encoding_failures)`` where:
      - ``face_targets`` is a list of dicts compatible with the existing
        face target / encoding plumbing (each has ``encoding`` and a
        synthetic ``person_id``).
      - ``encoding_failures`` is a list of ``{person_id, label, reason}``
        for images we could not decode or that contained no detectable
        face — surfaced back to the client so the editor can show a
        popup explaining what was skipped.
    """
    raw_value = data.get("face_images")
    if raw_value is None:
        raw_form = request.form.get("face_images", "")
        if raw_form:
            try:
                raw_value = json.loads(raw_form)
            except json.JSONDecodeError:
                raw_value = None
    if not isinstance(raw_value, list) or not raw_value:
        return [], []

    face_targets = []
    failures = []
    for index, entry in enumerate(raw_value):
        if not isinstance(entry, dict):
            continue
        person_id = str(entry.get("person_id") or "").strip()
        label = str(entry.get("label") or person_id or f"face_{index}").strip() or person_id
        image_b64 = entry.get("image_base64") or entry.get("snap_base64") or ""
        img = _decode_face_image_base64(image_b64)
        if img is None:
            failures.append({
                "person_id": person_id,
                "label": label,
                "reason": "Could not decode face image",
            })
            continue
        try:
            detections = detect_uploaded_reference_faces(img, with_encodings=True)
        except Exception as error:
            logger.warning("InsightFace failed on snapped face %s: %s", person_id or label, error)
            failures.append({
                "person_id": person_id,
                "label": label,
                "reason": "Face detector failed on the snapped image",
            })
            continue
        with_encoding = [
            det for det in (detections or [])
            if det.get("encoding") is not None
        ]
        if not with_encoding:
            failures.append({
                "person_id": person_id,
                "label": label,
                "reason": "No face encoding could be extracted from the snapped image",
            })
            continue
        # Pick the highest-scoring face — typically the snapped image is a
        # tight crop, so the largest/most-confident face is the right one.
        with_encoding.sort(
            key=lambda d: (
                float(d.get("det_score") or d.get("confidence") or 0.0),
                float(d.get("sharpness") or 0.0),
            ),
            reverse=True,
        )
        chosen = with_encoding[0]
        encoding = chosen.get("encoding")
        if encoding is None:
            failures.append({
                "person_id": person_id,
                "label": label,
                "reason": "Face encoding was empty",
            })
            continue
        try:
            encoding_list = np.asarray(encoding, dtype=np.float32).tolist()
        except (TypeError, ValueError):
            failures.append({
                "person_id": person_id,
                "label": label,
                "reason": "Could not normalize face encoding",
            })
            continue
        # Use the synthetic person_id from the editor so the snapped face
        # is identifiable in logs/responses; if absent, generate one.
        target_person_id = person_id or f"snap_{index}"
        face_targets.append({
            "person_id": target_person_id,
            "name": label,
            "description": label,
            "encoding": encoding_list,
            "is_snapped_entity": True,
        })

    return face_targets, failures


def build_redaction_request(data):
    job_id = data.get("job_id") or request.form.get("job_id")
    if not job_id:
        raise RedactionRequestError("job_id is required", 400)

    job = get_job(job_id)
    if not job:
        raise RedactionRequestError("job not found", 404)
    if job["status"] != "ready":
        raise RedactionRequestError(
            "job is not ready for redaction",
            409,
            {"status": job["status"]},
        )

    person_ids = parse_list_field(data, "person_ids", split_csv=True) or []
    person_ids = [str(item).strip() for item in person_ids if str(item).strip()]

    person_label_map_raw = data.get("person_labels")
    if person_label_map_raw is None:
        raw_label_form = request.form.get("person_labels", "")
        if raw_label_form:
            try:
                person_label_map_raw = json.loads(raw_label_form)
            except json.JSONDecodeError:
                person_label_map_raw = None
    person_label_map = {}
    if isinstance(person_label_map_raw, dict):
        for pid, label in person_label_map_raw.items():
            person_label_map[str(pid).strip()] = str(label).strip()

    face_encodings = parse_list_field(data, "face_encodings") or []
    face_targets = []
    matched_ids = []
    if person_ids:
        enriched = get_enriched_faces(job_id) or {}
        unique_faces = job.get("unique_faces") or enriched.get("unique_faces", [])
        for index, face in enumerate(unique_faces):
            stable_person_id = ensure_face_identity(face, fallback_index=index)
            if stable_person_id not in person_ids:
                continue
            face_targets.append(face)
            encoding = face.get("encoding")
            if encoding:
                face_encodings.append(encoding)
            matched_ids.append(stable_person_id)
        logger.info(
            "Resolved person_ids %s -> %d face targets (%s)",
            person_ids,
            len(face_targets),
            matched_ids,
        )

    # Snapped-face entities arrive via ``face_images`` (rather than via
    # the job's ``unique_faces``). Run them through InsightFace now so
    # their encodings join the per-frame relock pipeline like any other
    # selected face. Anything that produces no encoding becomes a
    # ``face_blur_failure`` we report back to the client.
    snapped_face_targets, snapped_face_failures = parse_face_images(data)
    for snap_face in snapped_face_targets:
        snap_person_id = str(snap_face.get("person_id") or "").strip()
        face_targets.append(snap_face)
        face_encodings.append(snap_face["encoding"])
        if snap_person_id and snap_person_id not in matched_ids:
            matched_ids.append(snap_person_id)

    # Person IDs that the editor sent but for which we found neither a
    # matching ``unique_faces`` entry nor a snapped image with a usable
    # encoding. These are surfaced back to the client so the editor can
    # tell the user which faces could not be redacted, instead of the
    # whole export silently failing with a 400.
    unresolved_person_ids = []
    for pid in person_ids:
        if pid in matched_ids:
            continue
        unresolved_person_ids.append({
            "person_id": pid,
            "label": person_label_map.get(pid) or pid,
            "reason": "No matching face encoding was available",
        })

    object_classes = parse_list_field(data, "object_classes", split_csv=True) or []
    object_classes = [str(item).strip() for item in object_classes if str(item).strip()]

    redaction_style = str(
        data.get("redaction_style", request.form.get("redaction_style", "blur")) or "blur"
    ).strip().lower()
    if redaction_style not in {"blur", "black"}:
        redaction_style = "blur"

    try:
        blur_strength = int(data.get("blur_strength", request.form.get("blur_strength", DEFAULT_BLUR_STRENGTH)))
    except (TypeError, ValueError):
        blur_strength = DEFAULT_BLUR_STRENGTH
    if redaction_style == "blur":
        blur_strength = max(DEFAULT_BLUR_STRENGTH, blur_strength)

    try:
        detect_every_n = int(data.get("detect_every_n", request.form.get("detect_every_n", DEFAULT_DETECT_EVERY_N)))
    except (TypeError, ValueError):
        detect_every_n = DEFAULT_DETECT_EVERY_N

    detect_every_seconds = data.get("detect_every_seconds")
    if detect_every_seconds is None:
        raw_seconds = request.form.get("detect_every_seconds", "")
        if raw_seconds:
            try:
                detect_every_seconds = float(raw_seconds)
            except (TypeError, ValueError):
                detect_every_seconds = None

    use_temporal_optimization = parse_bool_field(data, "use_temporal_optimization", default=True)
    output_height = parse_output_height(data)
    entity_ids = parse_list_field(data, "entity_ids", split_csv=True) or []
    entity_ids = [str(item).strip() for item in entity_ids if str(item).strip()]
    custom_regions = parse_custom_regions(data)

    has_face_targets = bool(face_targets or face_encodings)
    has_face_custom_regions = any(custom_region_is_face(region) for region in custom_regions)
    if has_face_targets or has_face_custom_regions:
        detect_every_n = 1

    total_targets = max(len(face_targets), len(face_encodings)) + len(object_classes) + len(custom_regions)
    if total_targets == 0 and not entity_ids:
        # Differentiate the "you selected nothing" case from the "you
        # selected only snapped faces but none of them had a usable
        # encoding" case so the editor can show a helpful popup instead
        # of a generic failure.
        if unresolved_person_ids or snapped_face_failures:
            raise RedactionRequestError(
                "None of the selected faces could be redacted: their face encodings could not be resolved.",
                400,
                {
                    "unresolved_person_ids": unresolved_person_ids,
                    "face_blur_failures": snapped_face_failures,
                },
            )
        raise RedactionRequestError(
            "No targets selected. Provide person_ids, face_encodings, object_classes, entity_ids, or custom_regions (drawn regions with motion tracking).",
            400,
        )

    return {
        "job_id": job_id,
        "person_ids": person_ids,
        "face_encodings": face_encodings,
        "face_targets": face_targets,
        "object_classes": object_classes,
        "entity_ids": entity_ids,
        "custom_regions": custom_regions,
        "blur_strength": blur_strength,
        "redaction_style": redaction_style,
        "detect_every_n": detect_every_n,
        "detect_every_seconds": detect_every_seconds,
        "use_temporal_optimization": use_temporal_optimization,
        "output_height": output_height,
        "person_label_map": person_label_map,
        "unresolved_person_ids": unresolved_person_ids,
        "face_blur_failures": snapped_face_failures,
    }


def serialize_redaction_response(prepared, result):
    return {
        "output_path": result["output_path"],
        "download_url": result["download_url"],
        "download_filename": result.get("download_filename"),
        "mime_type": result.get("mime_type", "video/mp4"),
        "output_size_bytes": result.get("output_size_bytes"),
        "h264_encoded": result.get("h264_encoded", False),
        "download_ready": result.get("download_ready", False),
        "export_quality": result.get("export_quality", f"{prepared.get('output_height', 720)}p"),
        "width": result.get("width"),
        "height": result.get("height"),
        "total_frames": result["total_frames"],
        "fps": result["fps"],
        "detection_frames_processed": result.get("detection_frames_processed", 0),
        "detection_frames_skipped": result.get("detection_frames_skipped", 0),
        "entity_ids_used": result.get("entity_ids_used", []),
        "temporal_ranges_from_entity_search": result.get("temporal_ranges_from_entity_search", 0),
        "person_ids_used": prepared.get("person_ids", []),
        "unresolved_person_ids": prepared.get("unresolved_person_ids", []),
        "face_blur_failures": prepared.get("face_blur_failures", []),
        "face_lock_failures": result.get("face_lock_failures", []),
    }


def update_redaction_job(redaction_job_id, **updates):
    with _redaction_jobs_lock:
        job = _redaction_jobs.get(redaction_job_id)
        if not job:
            return
        job.update(updates)


def run_redaction_job(redaction_job_id, prepared):
    def progress_callback(update):
        update_redaction_job(
            redaction_job_id,
            status="running",
            stage=update.get("stage") or "running",
            progress=update.get("progress", 0.0),
            percent=update.get("percent", 0),
            frames_processed=update.get("frames_processed", 0),
            total_frames=update.get("total_frames", 0),
            message=update.get("message"),
        )

    try:
        result = run_redaction(
            job_id=prepared["job_id"],
            face_encodings=prepared["face_encodings"],
            face_targets=prepared["face_targets"],
            object_classes=prepared["object_classes"],
            entity_ids=prepared["entity_ids"],
            custom_regions=prepared["custom_regions"],
            blur_strength=prepared["blur_strength"],
            redaction_style=prepared["redaction_style"],
            detect_every_n=prepared["detect_every_n"],
            detect_every_seconds=prepared["detect_every_seconds"],
            use_temporal_optimization=prepared["use_temporal_optimization"],
            output_height=prepared["output_height"],
            progress_callback=progress_callback,
        )
    except ObjectDetectionUnavailable as error:
        update_redaction_job(
            redaction_job_id,
            status="failed",
            stage="failed",
            error=str(error),
            error_status=503,
        )
        return
    except Exception as error:
        logger.exception("Redaction job %s failed", redaction_job_id)
        update_redaction_job(
            redaction_job_id,
            status="failed",
            stage="failed",
            error=str(error),
            error_status=500,
        )
        return

    payload = serialize_redaction_response(prepared, result)
    update_redaction_job(
        redaction_job_id,
        status="completed",
        stage="completed",
        progress=1.0,
        percent=100,
        frames_processed=result.get("total_frames", 0),
        total_frames=result.get("total_frames", 0),
        message="Redaction complete",
        result=payload,
    )


@redaction_bp.route("/redact", methods=["POST"])
def redact():
    data = request.get_json(silent=True) or {}
    try:
        prepared = build_redaction_request(data)
    except RedactionRequestError as error:
        return jsonify({"error": error.message, **error.payload}), error.status_code

    logger.info(
        "Redacting job %s: %d face targets / %d face encodings (person_ids=%s), %s object classes, %s entity_ids, %d custom_regions",
        prepared["job_id"],
        len(prepared["face_targets"]),
        len(prepared["face_encodings"]),
        prepared["person_ids"] or "none",
        prepared["object_classes"] or "none",
        len(prepared["entity_ids"]),
        len(prepared["custom_regions"]),
    )

    try:
        result = run_redaction(
            job_id=prepared["job_id"],
            face_encodings=prepared["face_encodings"],
            face_targets=prepared["face_targets"],
            object_classes=prepared["object_classes"],
            entity_ids=prepared["entity_ids"],
            custom_regions=prepared["custom_regions"],
            blur_strength=prepared["blur_strength"],
            redaction_style=prepared["redaction_style"],
            detect_every_n=prepared["detect_every_n"],
            detect_every_seconds=prepared["detect_every_seconds"],
            use_temporal_optimization=prepared["use_temporal_optimization"],
            output_height=prepared["output_height"],
        )
    except ObjectDetectionUnavailable as e:
        return jsonify({"error": str(e)}), 503

    return jsonify(serialize_redaction_response(prepared, result))


@redaction_bp.route("/redact/jobs", methods=["POST"])
def start_redaction():
    data = request.get_json(silent=True) or {}

    try:
        prepared = build_redaction_request(data)
    except RedactionRequestError as error:
        return jsonify({"error": error.message, **error.payload}), error.status_code

    redaction_job_id = uuid.uuid4().hex
    with _redaction_jobs_lock:
        _redaction_jobs[redaction_job_id] = {
            "redaction_job_id": redaction_job_id,
            "source_job_id": prepared["job_id"],
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "percent": 0,
            "frames_processed": 0,
            "total_frames": 0,
            "message": "Queued for rendering",
            "result": None,
            "error": None,
            "error_status": None,
        }

    thread = threading.Thread(
        target=run_redaction_job,
        args=(redaction_job_id, prepared),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "redaction_job_id": redaction_job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "percent": 0,
        "message": "Queued for rendering",
    }), 202


@redaction_bp.route("/redact/jobs/<redaction_job_id>", methods=["GET"])
def get_redaction_status(redaction_job_id):
    with _redaction_jobs_lock:
        job = dict(_redaction_jobs.get(redaction_job_id) or {})

    if not job:
        return jsonify({"error": "redaction job not found"}), 404

    payload = {
        "redaction_job_id": redaction_job_id,
        "source_job_id": job.get("source_job_id"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "progress": job.get("progress", 0.0),
        "percent": job.get("percent", 0),
        "frames_processed": job.get("frames_processed", 0),
        "total_frames": job.get("total_frames", 0),
        "message": job.get("message"),
        "error": job.get("error"),
    }
    if job.get("result") is not None:
        payload["result"] = job["result"]
    return jsonify(payload)


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
