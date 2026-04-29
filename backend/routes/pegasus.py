import logging

from flask import Blueprint, jsonify, request

from services import pegasus_privacy

logger = logging.getLogger("video_redaction.routes.pegasus")

pegasus_bp = Blueprint("pegasus", __name__)


def _service_error_status(error):
    message = str(error)
    if "TWELVELABS_API_KEY" in message or "not configured" in message:
        return 503
    if "timed out" in message.lower() or "timeout" in message.lower():
        return 504
    if "not found" in message.lower():
        return 404
    if "not ready" in message.lower():
        return 409
    return 500


@pegasus_bp.route("/pegasus/privacy-assist/jobs", methods=["POST"])
def create_privacy_assist_job():
    data = request.get_json(silent=True) or {}
    video_id = str(data.get("video_id") or "").strip()
    local_job_id = str(data.get("local_job_id") or "").strip() or None
    force = bool(data.get("force"))

    if not video_id:
        return jsonify({"error": "video_id is required"}), 400

    try:
        result = pegasus_privacy.start_privacy_assist_job(
            video_id,
            local_job_id=local_job_id,
            force=force,
        )
        status_code = 200 if result.get("cached") and result.get("status") == "ready" else 202
        return jsonify(result), status_code
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        logger.warning("Pegasus job create failed: %s", exc)
        return jsonify({"error": str(exc)}), _service_error_status(exc)
    except Exception as exc:
        logger.exception("Pegasus job create failed")
        return jsonify({"error": str(exc)}), _service_error_status(exc)


@pegasus_bp.route("/pegasus/privacy-assist/jobs/<job_id>", methods=["GET"])
def get_privacy_assist_job(job_id):
    try:
        return jsonify(pegasus_privacy.get_privacy_assist_job(job_id))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except RuntimeError as exc:
        logger.warning("Pegasus job poll failed: %s", exc)
        return jsonify({"error": str(exc)}), _service_error_status(exc)
    except Exception as exc:
        logger.exception("Pegasus job poll failed")
        return jsonify({"error": str(exc)}), _service_error_status(exc)


@pegasus_bp.route("/pegasus/privacy-assist/jobs/<job_id>/apply-preview", methods=["POST"])
def pegasus_apply_preview(job_id):
    data = request.get_json(silent=True) or {}
    local_job_id = str(data.get("local_job_id") or "").strip() or None

    try:
        return jsonify(pegasus_privacy.build_apply_preview(job_id, local_job_id=local_job_id))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    except RuntimeError as exc:
        logger.warning("Pegasus apply preview failed: %s", exc)
        return jsonify({"error": str(exc)}), _service_error_status(exc)
    except Exception as exc:
        logger.exception("Pegasus apply preview failed")
        return jsonify({"error": str(exc)}), _service_error_status(exc)
