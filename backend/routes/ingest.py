import logging
import os
import tempfile

from flask import Blueprint, request, jsonify

from config import OUTPUT_DIR
from services.pipeline import start_ingestion, get_job, list_jobs, push_job_entities_to_twelvelabs, get_job_id_by_video_id
from services import twelvelabs_service

logger = logging.getLogger("video_redaction.routes.index")

index_bp = Blueprint("index", __name__)


@index_bp.route("/index", methods=["POST"])
def index():
    if "video" not in request.files:
        return jsonify({"error": "missing video file"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "no file selected"}), 400

    try:
        interval = float(request.form.get("detect_interval_sec", 1.0))
    except (TypeError, ValueError):
        interval = 1.0

    skip_indexing = request.form.get("skip_indexing", "false").lower() in ("true", "1", "yes")
    video_id = (request.form.get("video_id") or "").strip()
    from_job_id = (request.form.get("from_job_id") or "").strip()
    if skip_indexing and from_job_id and not video_id:
        prev = get_job(from_job_id)
        if prev:
            video_id = (prev.get("twelvelabs_video_id") or "").strip()
        if not video_id:
            return jsonify({"error": "from_job_id has no twelvelabs_video_id; provide video_id instead"}), 400

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ext = os.path.splitext(video_file.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(
        dir=OUTPUT_DIR, suffix=ext, delete=False, prefix="upload_"
    )
    video_file.save(tmp.name)
    tmp.close()

    logger.info("Indexing video: %s (%s) skip_indexing=%s", video_file.filename, tmp.name, skip_indexing)

    job_id = start_ingestion(
        video_path=tmp.name,
        video_filename=video_file.filename,
        interval_sec=interval,
        skip_indexing=skip_indexing,
        existing_video_id=video_id or None,
    )

    return jsonify({
        "job_id": job_id,
        "status": "processing",
        "message": "Video index started. Poll /api/index/<job_id> for status.",
    }), 202


@index_bp.route("/index/<job_id>", methods=["GET"])
def index_status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "twelvelabs_status": job.get("twelvelabs_status"),
        "local_status": job.get("local_status"),
        "video_filename": job.get("video_filename"),
        "created_at": job.get("created_at"),
        "error": job.get("error"),
        "video_metadata": job.get("video_metadata"),
    })


@index_bp.route("/jobs", methods=["GET"])
def jobs():
    return jsonify({"jobs": list_jobs()})


@index_bp.route("/jobs/by-video/<video_id>", methods=["GET"])
def job_by_video(video_id):
    """Return job_id for the first job whose twelvelabs_video_id matches (for editor detection)."""
    job_id = get_job_id_by_video_id(video_id)
    if not job_id:
        return jsonify({"error": "no job found for this video"}), 404
    return jsonify({"job_id": job_id})


@index_bp.route("/jobs/<job_id>/push-entities", methods=["POST"])
def push_entities(job_id):
    """
    Explicit endpoint: push this job's face snaps to TwelveLabs as entities.
    Entities are only created when this endpoint is called; the pipeline never pushes automatically.
    """
    try:
        entities = push_job_entities_to_twelvelabs(job_id)
        return jsonify({
            "job_id": job_id,
            "pushed": True,
            "entities_count": len(entities),
            "entities": [{"entity_id": e.get("entity_id"), "name": e.get("name")} for e in entities],
        }), 200
    except AttributeError as e:
        logger.warning("Push entities failed (API not available): %s", e)
        return jsonify({
            "error": "TwelveLabs entity API (entity_collections) is not available in this SDK version.",
            "pushed": False,
        }), 503
    except ValueError as e:
        return jsonify({"error": str(e), "pushed": False}), 404 if "not found" in str(e).lower() else 400


@index_bp.route("/videos", methods=["GET"])
def list_videos():
    """List indexed videos from TwelveLabs (alias for /api/indexing/videos)."""
    index_id = request.args.get("index_id")
    page = int(request.args.get("page", 1))
    page_limit = int(request.args.get("page_limit", 10))
    try:
        result = twelvelabs_service.list_indexed_videos(
            index_id=index_id,
            page=page,
            page_limit=page_limit,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "videos": [], "index_id": index_id}), 200
