import logging
import os

from flask import Blueprint, Response, jsonify, request, send_file, stream_with_context

from config import OUTPUT_DIR, BASE_DIR
from utils.downloads import redacted_download_path
from utils.video import validate_mp4_output

logger = logging.getLogger("video_redaction.routes.download")

download_bp = Blueprint("download", __name__)

DOWNLOAD_CHUNK_SIZE = 1024 * 1024

GENERATED_THUMBNAILS_DIR = os.path.join(
    os.path.dirname(BASE_DIR), "frontend", "public", "generated-thumbnails"
)


def remove_download_after_response(path, filename):
    """Delete one-time redacted exports after the client response closes."""
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.info("Deleted one-time redacted download after serving: %s", filename)
    except OSError as error:
        logger.warning("Could not delete redacted download %s after serving: %s", filename, error)


def add_download_headers(response, filename, content_length=None):
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Cache-Control"] = "no-store, private, max-age=0, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    if content_length is not None:
        response.headers["Content-Length"] = str(content_length)
    return response


def stream_redacted_download_once(path, filename):
    """Stream a full redacted MP4 download and remove it when streaming ends."""
    content_length = os.path.getsize(path)

    def generate():
        try:
            with open(path, "rb") as video_file:
                while True:
                    chunk = video_file.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk
        finally:
            remove_download_after_response(path, filename)

    response = Response(
        stream_with_context(generate()),
        mimetype="video/mp4",
        direct_passthrough=True,
    )
    return add_download_headers(response, filename, content_length)


@download_bp.route("/download/<filename>", methods=["GET"])
def download(filename):
    try:
        safe, path = redacted_download_path(filename, OUTPUT_DIR)
    except ValueError:
        return jsonify({"error": "only redacted mp4 downloads are supported"}), 400
    if not os.path.isfile(path):
        return jsonify({"error": "file not found"}), 404
    try:
        validate_mp4_output(path)
    except ValueError as error:
        logger.warning("Refusing invalid redacted MP4 download %s: %s", safe, error)
        return jsonify({"error": "redacted mp4 is not ready for download"}), 410

    if request.method == "GET" and not request.headers.get("Range"):
        return stream_redacted_download_once(path, safe)

    response = send_file(
        path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=safe,
        conditional=True,
        max_age=0,
    )
    return add_download_headers(response, safe)


@download_bp.route("/thumbnails/<video_id>.jpg", methods=["GET"])
def serve_thumbnail(video_id):
    safe = os.path.basename(video_id)
    path = os.path.join(GENERATED_THUMBNAILS_DIR, f"{safe}.jpg")
    if not os.path.isfile(path):
        return jsonify({"error": "thumbnail not found"}), 404
    return send_file(path, mimetype="image/jpeg")
