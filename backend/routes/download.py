import logging
import os

from flask import Blueprint, jsonify, send_file

from config import OUTPUT_DIR

logger = logging.getLogger("video_redaction.routes.download")

download_bp = Blueprint("download", __name__)


@download_bp.route("/download/<filename>", methods=["GET"])
def download(filename):
    safe = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "file not found"}), 404
    return send_file(
        path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=safe,
        conditional=False,
        max_age=0,
    )
