import logging

from flask import Blueprint, request, jsonify

from services import twelvelabs_service

logger = logging.getLogger("video_redaction.routes.search")

search_bp = Blueprint("search", __name__)


@search_bp.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = data.get("query")
    index_id = data.get("index_id")

    if not query:
        return jsonify({"error": "query is required"}), 400

    results = twelvelabs_service.search_segments(query, index_id=index_id)
    return jsonify({
        "query": query,
        "index_id": index_id or twelvelabs_service.get_index_id(),
        "results": results,
    })


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
