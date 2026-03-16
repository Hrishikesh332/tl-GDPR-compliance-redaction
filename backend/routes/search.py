import json
import logging
import os
import tempfile

from flask import Blueprint, request, jsonify

from config import OUTPUT_DIR
from services import twelvelabs_service

logger = logging.getLogger("video_redaction.routes.search")

search_bp = Blueprint("search", __name__)


@search_bp.route("/search", methods=["POST"])
def search():
    query = (request.form.get("query") or "").strip() or None
    index_id = (request.form.get("index_id") or "").strip() or None
    image_url = (request.form.get("image_url") or "").strip() or None
    operator = (request.form.get("operator") or "").strip().lower() or None

    if not query and not index_id and not image_url and not operator and "image" not in request.files:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip() or None
        index_id = (data.get("index_id") or "").strip() or None
        image_url = (data.get("image_url") or "").strip() or None
        operator = (data.get("operator") or "").strip().lower() or None

    if operator not in {"and", "or"}:
        operator = None

    image_files = [file for file in request.files.getlist("image") if file]
    if not query and not image_url and not image_files:
        return jsonify({"error": "query text or image is required"}), 400

    try:
        if image_files:
            tmp_paths = []
            for image_file in image_files:
                if not image_file.filename:
                    return jsonify({"error": "no image selected"}), 400

                os.makedirs(OUTPUT_DIR, exist_ok=True)
                ext = os.path.splitext(image_file.filename)[1] or ".png"
                tmp = tempfile.NamedTemporaryFile(
                    dir=OUTPUT_DIR, suffix=ext, delete=False, prefix="search_image_"
                )
                image_file.save(tmp.name)
                tmp.close()
                tmp_paths.append(tmp.name)
            try:
                results = twelvelabs_service.search_segments(
                    query=query,
                    index_id=index_id,
                    image_paths=tmp_paths,
                    operator=operator,
                )
            finally:
                for tmp_path in tmp_paths:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        else:
            results = twelvelabs_service.search_segments(
                query=query,
                index_id=index_id,
                image_url=image_url,
                operator=operator,
            )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Search request failed")
        return jsonify({"error": str(e)}), 500

    payload = {
        "query": query,
        "index_id": index_id or twelvelabs_service.get_index_id(),
        "group_by": "video",
        "operator": operator,
        "results": results,
    }
    logger.info("Search response payload: %s", json.dumps(payload, default=str, indent=2))
    return jsonify(payload)


@search_bp.route("/search/person-segments", methods=["POST"])
def person_segments():
    data = request.get_json(force=True)
    video_id = data.get("video_id")
    description = data.get("description")

    if not description:
        return jsonify({"error": "description is required"}), 400

    ranges = twelvelabs_service.find_person_time_ranges(video_id, description)
    return jsonify({
        "video_id": video_id,
        "description": description,
        "time_ranges": ranges,
    })
