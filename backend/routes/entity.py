import logging
import os
import tempfile

from flask import Blueprint, request, jsonify

from config import OUTPUT_DIR
from services import twelvelabs_service

logger = logging.getLogger("video_redaction.routes.entity")

entity_bp = Blueprint("entity", __name__)

ENTITY_UNAVAILABLE = (
    "TwelveLabs entity API (entity_collections) is not available in this SDK version. "
    "Use face encodings from job faces for redaction instead."
)


def _entity_api_available():
    try:
        client = twelvelabs_service._get_client()
        return hasattr(client, "entity_collections") and client.entity_collections is not None
    except Exception:
        return False


@entity_bp.route("/entities", methods=["GET"])
def list_entities():
    if not _entity_api_available():
        return jsonify({
            "entity_collection_id": None,
            "entities": [],
            "unavailable": True,
            "message": ENTITY_UNAVAILABLE,
        })
    try:
        entities = twelvelabs_service.list_entities()
        return jsonify({
            "entity_collection_id": twelvelabs_service.get_entity_collection_id(),
            "entities": entities,
        })
    except AttributeError as e:
        logger.warning("Entity API not available: %s", e)
        return jsonify({
            "entity_collection_id": None,
            "entities": [],
            "unavailable": True,
            "message": ENTITY_UNAVAILABLE,
        })


@entity_bp.route("/entities/<entity_id>", methods=["GET"])
def get_entity(entity_id):
    if not _entity_api_available():
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True}), 503
    try:
        entity = twelvelabs_service.retrieve_entity(entity_id)
        return jsonify(entity)
    except (AttributeError, Exception) as e:
        return jsonify({"error": str(e)}), 404 if "not found" in str(e).lower() else 503


@entity_bp.route("/entities", methods=["POST"])
def create_entity():
    if not _entity_api_available():
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True}), 503
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    asset_ids = data.get("asset_ids")
    description = data.get("description")
    metadata = data.get("metadata")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not asset_ids or not isinstance(asset_ids, list):
        return jsonify({"error": "asset_ids (list) is required"}), 400

    try:
        result = twelvelabs_service.create_entity(
            name=name,
            asset_ids=asset_ids,
            description=description,
            metadata=metadata,
        )
        return jsonify(result), 201
    except AttributeError as e:
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True}), 503


@entity_bp.route("/entities/<entity_id>", methods=["DELETE"])
def delete_entity(entity_id):
    if not _entity_api_available():
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True}), 503
    try:
        twelvelabs_service.delete_entity(entity_id)
        return jsonify({"deleted": entity_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@entity_bp.route("/entities/upload-face", methods=["POST"])
def upload_face_and_create_entity():
    if not _entity_api_available():
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True}), 503
    if "image" not in request.files:
        return jsonify({"error": "missing image file"}), 400

    image_file = request.files["image"]
    name = request.form.get("name", "")
    description = request.form.get("description", "")
    preview_base64 = (request.form.get("preview_base64") or "").strip()

    if not name:
        return jsonify({"error": "name is required"}), 400

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ext = os.path.splitext(image_file.filename)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(
        dir=OUTPUT_DIR, suffix=ext, delete=False, prefix="entity_face_"
    )
    image_file.save(tmp.name)
    tmp.close()

    try:
        asset_id = twelvelabs_service.upload_face_asset(tmp.name)
        metadata = {"name": name}
        if preview_base64:
            metadata["face_snap_base64"] = preview_base64

        entity_result = twelvelabs_service.create_entity(
            name=name,
            asset_ids=[asset_id],
            description=description or f"Face entity: {name}",
            metadata=metadata,
        )

        return jsonify({
            "asset_id": asset_id,
            "entity": entity_result,
        }), 201
    except AttributeError:
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True}), 503
    except Exception as e:
        logger.exception("upload_face_and_create_entity failed")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


@entity_bp.route("/entities/<entity_id>/add-asset", methods=["POST"])
def add_asset_to_entity(entity_id):
    if not _entity_api_available():
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True}), 503
    if "image" in request.files:
        image_file = request.files["image"]
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ext = os.path.splitext(image_file.filename)[1] or ".jpg"
        tmp = tempfile.NamedTemporaryFile(
            dir=OUTPUT_DIR, suffix=ext, delete=False, prefix="entity_asset_"
        )
        image_file.save(tmp.name)
        tmp.close()
        try:
            asset_id = twelvelabs_service.upload_face_asset(tmp.name)
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
    else:
        data = request.get_json(silent=True) or {}
        image_url = data.get("image_url")
        if not image_url:
            return jsonify({"error": "provide image file or image_url"}), 400
        asset_id = twelvelabs_service.upload_face_asset_from_url(image_url)

    try:
        client = twelvelabs_service._get_client()
        collection_id = twelvelabs_service.get_entity_collection_id()

        existing = client.entity_collections.entities.retrieve(
            entity_collection_id=collection_id,
            entity_id=entity_id,
        )
        current_assets = list(existing.asset_ids or [])
        current_assets.append(asset_id)

        client.entity_collections.entities.update(
            entity_collection_id=collection_id,
            entity_id=entity_id,
        )

        return jsonify({
            "entity_id": entity_id,
            "new_asset_id": asset_id,
            "total_assets": len(current_assets),
        })
    except AttributeError:
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True}), 503


@entity_bp.route("/entities/<entity_id>/search", methods=["POST"])
def search_by_entity(entity_id):
    if not _entity_api_available():
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True, "results": []}), 503
    data = request.get_json(silent=True) or {}
    query_suffix = data.get("query", "")
    index_id = data.get("index_id")

    try:
        results = twelvelabs_service.entity_search(
            entity_id=entity_id,
            query_suffix=query_suffix,
            index_id=index_id,
        )
        return jsonify({
            "entity_id": entity_id,
            "query_suffix": query_suffix,
            "index_id": index_id or twelvelabs_service.get_index_id(),
            "results": results,
        })
    except AttributeError:
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True, "results": []}), 503


@entity_bp.route("/entities/<entity_id>/time-ranges", methods=["POST"])
def entity_time_ranges(entity_id):
    if not _entity_api_available():
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True, "time_ranges": []}), 503
    data = request.get_json(silent=True) or {}
    video_id = data.get("video_id")
    index_id = data.get("index_id")

    try:
        ranges = twelvelabs_service.entity_search_time_ranges(
            entity_id=entity_id,
            video_id=video_id,
            index_id=index_id,
        )
        return jsonify({
            "entity_id": entity_id,
            "video_id": video_id,
            "time_ranges": ranges,
        })
    except AttributeError:
        return jsonify({"error": ENTITY_UNAVAILABLE, "unavailable": True, "time_ranges": []}), 503


@entity_bp.route("/entity-collections", methods=["GET"])
def list_collections():
    if not _entity_api_available():
        return jsonify({"collections": [], "unavailable": True, "message": ENTITY_UNAVAILABLE})
    try:
        collections = twelvelabs_service.list_entity_collections()
        return jsonify({"collections": collections})
    except AttributeError:
        return jsonify({"collections": [], "unavailable": True, "message": ENTITY_UNAVAILABLE})
