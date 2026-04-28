"""HTTP endpoints for the face-lock track builder.

Provides:
- POST /api/face-lock-track/build  -> queue a background build
- GET  /api/face-lock-track/<job_id>/<person_id>  -> return lane + status

The lane is the precomputed per-frame bbox track for a single person and
is the source of truth for both the live preview overlay and the export
render. The build is heavy enough to warrant a background thread, so the
POST returns immediately with a status, and clients poll the GET route
until ``status == "ready"``.
"""

import logging
import threading

from flask import Blueprint, request, jsonify

from services.face_lock_track import (
    build_face_lock_lane,
    get_face_lock_build_status,
    get_face_lock_lane,
)

logger = logging.getLogger("video_redaction.routes.face_lock")

face_lock_bp = Blueprint("face_lock", __name__)

_active_builds = {}
_active_builds_lock = threading.Lock()


def _build_key(job_id, person_id):
    return f"{job_id}:{person_id}"


def _run_build(job_id, person_id, force_rebuild):
    key = _build_key(job_id, person_id)
    try:
        build_face_lock_lane(job_id, person_id, force_rebuild=force_rebuild)
    except Exception as exc:
        logger.exception("Face-lock build failed for job=%s person=%s", job_id, person_id)
        # The builder writes "running" state into the shared dict; on
        # failure we record a failed status that the GET route will
        # surface to the client.
        from services.face_lock_track import _set_build_state  # local import keeps the public surface clean
        _set_build_state(
            job_id, person_id,
            status="failed", progress=0.0, percent=0, message=str(exc) or "build failed",
        )
    finally:
        with _active_builds_lock:
            _active_builds.pop(key, None)


@face_lock_bp.route("/face-lock-track/build", methods=["POST"])
def start_face_lock_build():
    data = request.get_json(silent=True) or {}
    job_id = str(data.get("job_id") or request.form.get("job_id") or "").strip()
    person_id = str(data.get("person_id") or request.form.get("person_id") or "").strip()
    force_rebuild = bool(data.get("force_rebuild") or False)

    if not job_id:
        return jsonify({"error": "job_id is required"}), 400
    if not person_id:
        return jsonify({"error": "person_id is required"}), 400

    if not force_rebuild:
        cached = get_face_lock_lane(job_id, person_id)
        if cached is not None:
            return jsonify({
                "status": "ready",
                "progress": 1.0,
                "percent": 100,
                "job_id": job_id,
                "person_id": person_id,
                "cached": True,
            })

    key = _build_key(job_id, person_id)
    with _active_builds_lock:
        existing = _active_builds.get(key)
        if existing is not None and existing.is_alive() and not force_rebuild:
            status = get_face_lock_build_status(job_id, person_id)
            return jsonify({
                "status": status.get("status", "running"),
                "progress": status.get("progress", 0.0),
                "percent": status.get("percent", 0),
                "job_id": job_id,
                "person_id": person_id,
                "queued": False,
            })

        thread = threading.Thread(
            target=_run_build,
            args=(job_id, person_id, force_rebuild),
            daemon=True,
        )
        _active_builds[key] = thread
        thread.start()

    return jsonify({
        "status": "queued",
        "progress": 0.0,
        "percent": 0,
        "job_id": job_id,
        "person_id": person_id,
        "queued": True,
    }), 202


@face_lock_bp.route("/face-lock-track/<job_id>/<person_id>", methods=["GET"])
def get_face_lock_lane_route(job_id, person_id):
    job_id = str(job_id or "").strip()
    person_id = str(person_id or "").strip()
    if not job_id or not person_id:
        return jsonify({"error": "job_id and person_id are required"}), 400

    include_lane = str(request.args.get("include_lane", "true")).lower() not in ("false", "0", "no")

    lane = get_face_lock_lane(job_id, person_id)
    status_info = get_face_lock_build_status(job_id, person_id)

    payload = {
        "job_id": job_id,
        "person_id": person_id,
        "status": "ready" if lane is not None else status_info.get("status", "missing"),
        "progress": status_info.get("progress", 1.0 if lane is not None else 0.0),
        "percent": status_info.get("percent", 100 if lane is not None else 0),
        "message": status_info.get("message"),
        "cached": lane is not None,
    }
    if lane is not None and include_lane:
        payload["lane"] = lane
    elif lane is not None:
        payload["video"] = lane.get("video")
        payload["segments"] = lane.get("segments")
        payload["safety_pad_ratio"] = lane.get("safety_pad_ratio")

    if lane is None and status_info.get("status") == "failed":
        return jsonify(payload), 500
    return jsonify(payload)
