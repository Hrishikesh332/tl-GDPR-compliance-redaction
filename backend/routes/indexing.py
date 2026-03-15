import logging
import os
import tempfile
import threading

from flask import Blueprint, request, jsonify

from config import OUTPUT_DIR, TWELVELABS_INDEX_ID  # Fixed index for all indexing (from .env)
from services import twelvelabs_service

logger = logging.getLogger("video_redaction.routes.indexing")

indexing_bp = Blueprint("indexing", __name__)

_indexing_jobs = {}
_lock = threading.Lock()


def track_indexing_async(tracking_id, task_id):
    def _run():
        try:
            with _lock:
                _indexing_jobs[tracking_id]["status"] = "indexing"

            result = twelvelabs_service.wait_for_indexing(task_id)

            with _lock:
                _indexing_jobs[tracking_id]["status"] = result["status"]
                _indexing_jobs[tracking_id]["video_id"] = result["video_id"]
                _indexing_jobs[tracking_id]["result"] = result
        except Exception as e:
            logger.error("Indexing tracking failed for %s: %s", tracking_id, str(e))
            with _lock:
                _indexing_jobs[tracking_id]["status"] = "failed"
                _indexing_jobs[tracking_id]["error"] = str(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


@indexing_bp.route("/indexing", methods=["POST"])
def index_video():
    if "video" in request.files:
        video_file = request.files["video"]
        if video_file.filename == "":
            return jsonify({"error": "no file selected"}), 400

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ext = os.path.splitext(video_file.filename)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(
            dir=OUTPUT_DIR, suffix=ext, delete=False, prefix="index_"
        )
        video_file.save(tmp.name)
        tmp.close()

        logger.info("Indexing uploaded video: %s -> %s (index=%s)", video_file.filename, tmp.name, TWELVELABS_INDEX_ID)
        result = twelvelabs_service.index_video_from_file(tmp.name, index_id=TWELVELABS_INDEX_ID)
        result["source"] = "upload"
        result["filename"] = video_file.filename

    else:
        data = request.get_json(silent=True) or {}
        video_url = data.get("video_url")
        video_path = data.get("video_path")

        if video_url:
            logger.info("Indexing video from URL: %s (index=%s)", video_url, TWELVELABS_INDEX_ID)
            result = twelvelabs_service.index_video_from_url(video_url, index_id=TWELVELABS_INDEX_ID)
            result["source"] = "url"

        elif video_path:
            if not os.path.isfile(video_path):
                return jsonify({"error": f"file not found: {video_path}"}), 400
            logger.info("Indexing local video: %s (index=%s)", video_path, TWELVELABS_INDEX_ID)
            result = twelvelabs_service.index_video_from_file(video_path, index_id=TWELVELABS_INDEX_ID)
            result["source"] = "local"
            result["video_path"] = video_path

        else:
            return jsonify({"error": "provide video file, video_url, or video_path"}), 400

    tracking_id = result["task_id"]
    with _lock:
        _indexing_jobs[tracking_id] = {
            "task_id": result["task_id"],
            "video_id": result["video_id"],
            "index_id": result["index_id"],
            "status": "submitted",
            "source": result.get("source"),
            "error": None,
            "result": None,
        }

    wait = request.args.get("wait", "false").lower() == "true"
    if not wait:
        track_indexing_async(tracking_id, result["task_id"])

        return jsonify({
            "task_id": result["task_id"],
            "video_id": result["video_id"],
            "index_id": result["index_id"],
            "status": "submitted",
            "source": result.get("source"),
            "message": "Indexing started. Poll /api/indexing/tasks/<task_id> for status.",
        }), 202

    logger.info("Waiting synchronously for indexing task %s", result["task_id"])
    final = twelvelabs_service.wait_for_indexing(result["task_id"])
    return jsonify(final)


@indexing_bp.route("/indexing/local", methods=["POST"])
def index_local_file():
    data = request.get_json(force=True)
    video_path = data.get("video_path")

    if not video_path:
        return jsonify({"error": "video_path is required"}), 400
    if not os.path.isfile(video_path):
        return jsonify({"error": f"file not found: {video_path}"}), 400

    logger.info("Indexing local file: %s (index=%s)", video_path, TWELVELABS_INDEX_ID)
    result = twelvelabs_service.index_video_from_file(video_path, index_id=TWELVELABS_INDEX_ID)

    tracking_id = result["task_id"]
    with _lock:
        _indexing_jobs[tracking_id] = {
            "task_id": result["task_id"],
            "video_id": result["video_id"],
            "index_id": result["index_id"],
            "status": "submitted",
            "source": "local",
            "error": None,
            "result": None,
        }

    track_indexing_async(tracking_id, result["task_id"])

    return jsonify({
        "task_id": result["task_id"],
        "video_id": result["video_id"],
        "index_id": result["index_id"],
        "status": "submitted",
        "video_path": video_path,
        "message": "Indexing started. Poll /api/indexing/tasks/<task_id> for status.",
    }), 202


@indexing_bp.route("/indexing/tasks", methods=["GET"])
def list_tasks():
    index_id = request.args.get("index_id")
    status_filter = request.args.get("status")
    page = int(request.args.get("page", 1))
    page_limit = int(request.args.get("page_limit", 10))

    if status_filter:
        status_filter = [s.strip() for s in status_filter.split(",")]

    result = twelvelabs_service.list_indexing_tasks(
        index_id=index_id,
        status_filter=status_filter,
        page=page,
        page_limit=page_limit,
    )
    return jsonify(result)


@indexing_bp.route("/indexing/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    with _lock:
        local_job = _indexing_jobs.get(task_id)

    tl_status = twelvelabs_service.get_task_status(task_id)

    response = {**tl_status}
    if local_job:
        response["tracking_status"] = local_job["status"]
        response["error"] = local_job.get("error")

    return jsonify(response)


@indexing_bp.route("/indexing/videos", methods=["GET"])
def list_videos():
    index_id = request.args.get("index_id")
    page = int(request.args.get("page", 1))
    page_limit = int(request.args.get("page_limit", 10))

    result = twelvelabs_service.list_indexed_videos(
        index_id=index_id,
        page=page,
        page_limit=page_limit,
    )
    return jsonify(result)


@indexing_bp.route("/indexing/videos/<video_id>", methods=["GET"])
def get_video(video_id):
    index_id = request.args.get("index_id")
    try:
        result = twelvelabs_service.get_video_info(video_id, index_id=index_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@indexing_bp.route("/indexing/videos/<video_id>", methods=["DELETE"])
def delete_video(video_id):
    index_id = request.args.get("index_id")
    try:
        twelvelabs_service.delete_indexed_video(video_id, index_id=index_id)
        return jsonify({"deleted": video_id, "index_id": index_id or TWELVELABS_INDEX_ID})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@indexing_bp.route("/indexing/info", methods=["GET"])
def index_info():
    index_id = request.args.get("index_id")
    try:
        result = twelvelabs_service.get_index_info(index_id=index_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
