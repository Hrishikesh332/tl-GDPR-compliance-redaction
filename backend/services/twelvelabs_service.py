import json
import logging
import time

from twelvelabs import TwelveLabs

from config import (
    TWELVELABS_API_KEY, TWELVELABS_INDEX_ID,
    TWELVELABS_ENTITY_COLLECTION_ID, TWELVELABS_ENTITY_COLLECTION_NAME,
)

logger = logging.getLogger("video_redaction.twelvelabs")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = TwelveLabs(api_key=TWELVELABS_API_KEY)
    return _client


def get_index_id():
    return TWELVELABS_INDEX_ID


def ingest_video(video_path, callback=None):
    client = _get_client()
    logger.info("Uploading video to TwelveLabs index %s", TWELVELABS_INDEX_ID)

    task = client.tasks.create(
        index_id=TWELVELABS_INDEX_ID,
        video_file=open(video_path, "rb"),
        enable_video_stream=True,
    )
    logger.info("Created task %s, video_id=%s", task.id, task.video_id)

    def _on_update(t):
        logger.info("Indexing status: %s", t.status)
        if callback:
            callback(t.status)

    completed = client.tasks.wait_for_done(
        task_id=task.id,
        sleep_interval=5.0,
        callback=_on_update,
    )

    if completed.status != "ready":
        raise RuntimeError(f"Indexing failed with status {completed.status}")

    logger.info("Indexing complete. video_id=%s", completed.video_id)
    return {
        "task_id": task.id,
        "video_id": completed.video_id,
        "status": completed.status,
    }


def get_task_status(task_id):
    client = _get_client()
    task = client.tasks.retrieve(task_id=task_id)
    return {
        "task_id": task.id,
        "video_id": task.video_id,
        "status": task.status,
        "index_id": task.index_id,
    }


def describe_people(video_id):
    client = _get_client()
    logger.info("Analyzing people in video %s", video_id)

    result = client.analyze(
        video_id=video_id,
        prompt=(
            "List every distinct person visible in this video. "
            "For each person, provide: "
            "1) Their name if visible (from name tags, captions, chyrons, or on-screen text). "
            "   If the name is not identifiable, leave it blank. "
            "2) A brief physical description (hair, clothing, distinguishing features) "
            "3) The approximate time ranges (start and end in seconds) when they are visible. "
            "Return as a JSON array with objects having keys: "
            "name, description, time_ranges (array of {start_sec, end_sec})."
        ),
        temperature=0.1,
    )

    try:
        text = result.data or ""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse people descriptions as JSON, returning raw text")
        return {"raw_description": result.data}


def describe_objects(video_id):
    client = _get_client()
    logger.info("Analyzing objects in video %s", video_id)

    result = client.analyze(
        video_id=video_id,
        prompt=(
            "You are analyzing this video for a multi-source legal investigation. "
            "List all objects and items that could be relevant as evidence: "
            "vehicles (cars, trucks, motorcycles), weapons (knives, guns), "
            "personal items (phones, bags, wallets, documents, IDs), "
            "electronics (laptops, monitors), clothing, and any other "
            "forensically significant items. "
            "For each object, provide: "
            "1) Object name/type "
            "2) A brief description including color, brand if visible, condition "
            "3) The approximate time ranges (start and end in seconds) when they are visible. "
            "Return as a JSON array with objects having keys: "
            "object_name, description, time_ranges (array of {start_sec, end_sec})."
        ),
        temperature=0.1,
    )

    try:
        text = result.data or ""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse object descriptions as JSON, returning raw text")
        return {"raw_description": result.data}


def get_scene_summary(video_id):
    client = _get_client()
    logger.info("Getting scene summary for video %s", video_id)

    result = client.analyze(
        video_id=video_id,
        prompt=(
            "Provide a detailed scene-by-scene summary of this video. "
            "Include scene changes, camera angles, lighting conditions, "
            "and any notable transitions. Return as JSON with key 'scenes' "
            "containing an array of {start_sec, end_sec, description}."
        ),
        temperature=0.1,
    )

    try:
        text = result.data or ""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw_summary": result.data}


def _raw_response_to_loggable(raw_list):
    """Convert raw API response items to JSON-serializable dicts for logging."""
    out = []
    for item in raw_list:
        clips = getattr(item, "clips", None)
        if clips is not None:
            out.append({
                "id": getattr(item, "id", None),
                "score": getattr(item, "score", None),
                "clips": [
                    {
                        "start": getattr(c, "start", None),
                        "end": getattr(c, "end", None),
                        "score": getattr(c, "score", None),
                        "rank": getattr(c, "rank", None),
                        "thumbnail_url": getattr(c, "thumbnail_url", None),
                    }
                    for c in clips
                ],
            })
        else:
            out.append({
                "video_id": getattr(item, "video_id", None),
                "id": getattr(item, "id", None),
                "start": getattr(item, "start", None),
                "end": getattr(item, "end", None),
                "score": getattr(item, "score", None),
                "rank": getattr(item, "rank", None),
                "thumbnail_url": getattr(item, "thumbnail_url", None),
            })
    return out


def _serialize_search_results(response):
    results = []
    for item in response:
        grouped_clips = getattr(item, "clips", None)
        grouped_video_id = getattr(item, "id", None)
        if grouped_video_id and grouped_clips:
            video_result = {
                "video_id": grouped_video_id,
                "score": getattr(item, "score", None),
                "clips": [],
            }
            for clip in grouped_clips:
                video_result["clips"].append({
                    "start": clip.start,
                    "end": clip.end,
                    "score": getattr(clip, "score", None),
                    "rank": getattr(clip, "rank", None),
                    "thumbnail_url": getattr(clip, "thumbnail_url", None),
                })
            results.append(video_result)
            continue

        results.append({
            "video_id": getattr(item, "video_id", None) or grouped_video_id,
            "start": getattr(item, "start", None),
            "end": getattr(item, "end", None),
            "score": getattr(item, "score", None),
            "rank": getattr(item, "rank", None),
            "thumbnail_url": getattr(item, "thumbnail_url", None),
        })

    return results


def search_segments(query=None, index_id=None, image_path=None, image_url=None):
    client = _get_client()
    idx = index_id or TWELVELABS_INDEX_ID
    if not query and not image_path and not image_url:
        raise ValueError("query text or image is required")

    logger.info(
        "Searching index %s (text=%s, image_path=%s, image_url=%s)",
        idx,
        bool(query),
        bool(image_path),
        bool(image_url),
    )

    kwargs = {
        "index_id": idx,
        "search_options": ["visual"],
        "group_by": "video",
        "page_limit": 20,
        "sort_option": "score",
    }
    if query:
        kwargs["query_text"] = query

    if image_url:
        response = client.search.query(
            **kwargs,
            query_media_type="image",
            query_media_url=image_url,
        )
        raw_list = list(response)
        logger.info("Search raw API response (%d items): %s", len(raw_list), json.dumps(_raw_response_to_loggable(raw_list), default=str, indent=2))
        return _serialize_search_results(iter(raw_list))

    if image_path:
        with open(image_path, "rb") as image_file:
            response = client.search.query(
                **kwargs,
                query_media_type="image",
                query_media_file=image_file,
            )
        raw_list = list(response)
        logger.info("Search raw API response (%d items): %s", len(raw_list), json.dumps(_raw_response_to_loggable(raw_list), default=str, indent=2))
        return _serialize_search_results(iter(raw_list))

    response = client.search.query(**kwargs)
    raw_list = list(response)
    logger.info("Search raw API response (%d items): %s", len(raw_list), json.dumps(_raw_response_to_loggable(raw_list), default=str, indent=2))
    return _serialize_search_results(iter(raw_list))


def find_person_time_ranges(video_id, person_description):
    client = _get_client()
    logger.info("Finding time ranges for person: %s", person_description)

    response = client.search.query(
        index_id=TWELVELABS_INDEX_ID,
        search_options=["visual"],
        query_text=person_description,
        page_limit=50,
    )

    ranges = []
    for item in response:
        if item.video_id == video_id or not video_id:
            ranges.append({
                "start": item.start,
                "end": item.end,
            })

    return ranges


# Instruction added to every analyze request so responses use markdown and mm:ss timestamps
ANALYZE_FORMAT_INSTRUCTION = (
    "Format your response in clear markdown: use **bold** for emphasis, bullet or numbered lists "
    "where appropriate, and headings (##) for sections. When referencing specific moments in the "
    "video, include timestamps in mm:ss format (e.g. 02:30 or 1:05)."
)


def analyze_video_custom(video_id, prompt):
    client = _get_client()
    logger.info("Custom analysis on video %s", video_id)
    enhanced_prompt = f"{ANALYZE_FORMAT_INSTRUCTION}\n\n{prompt}"
    result = client.analyze(video_id=video_id, prompt=enhanced_prompt, temperature=0.2)
    return {"data": result.data, "id": result.id}


def index_video_from_file(video_path, index_id=None, enable_stream=True):
    client = _get_client()
    idx = index_id or TWELVELABS_INDEX_ID
    logger.info("Submitting indexing task for %s -> index %s", video_path, idx)

    task = client.tasks.create(
        index_id=idx,
        video_file=open(video_path, "rb"),
        enable_video_stream=enable_stream,
    )

    logger.info("Indexing task created: task_id=%s, video_id=%s", task.id, task.video_id)
    return {
        "task_id": task.id,
        "video_id": task.video_id,
        "index_id": idx,
        "status": "submitted",
    }


def index_video_from_url(video_url, index_id=None, enable_stream=True):
    client = _get_client()
    idx = index_id or TWELVELABS_INDEX_ID
    logger.info("Submitting indexing task for URL %s -> index %s", video_url, idx)

    task = client.tasks.create(
        index_id=idx,
        video_url=video_url,
        enable_video_stream=enable_stream,
    )

    logger.info("Indexing task created: task_id=%s, video_id=%s", task.id, task.video_id)
    return {
        "task_id": task.id,
        "video_id": task.video_id,
        "index_id": idx,
        "status": "submitted",
    }


def wait_for_indexing(task_id, callback=None):
    client = _get_client()

    def _on_update(t):
        logger.info("Task %s: status=%s", task_id, t.status)
        if callback:
            callback(t.status)

    completed = client.tasks.wait_for_done(
        task_id=task_id,
        sleep_interval=5.0,
        callback=_on_update,
    )

    return {
        "task_id": completed.id,
        "video_id": completed.video_id,
        "status": completed.status,
        "index_id": completed.index_id,
    }


def list_indexing_tasks(index_id=None, status_filter=None, page=1, page_limit=10):
    client = _get_client()
    idx = index_id or TWELVELABS_INDEX_ID

    kwargs = {
        "page": page,
        "page_limit": page_limit,
        "index_id": idx,
        "sort_by": "created_at",
        "sort_option": "desc",
    }
    if status_filter:
        kwargs["status"] = status_filter

    response = client.tasks.list(**kwargs)

    tasks = []
    for task in response:
        meta = {}
        if task.system_metadata:
            meta = {
                "filename": getattr(task.system_metadata, "filename", None),
                "duration": getattr(task.system_metadata, "duration", None),
                "width": getattr(task.system_metadata, "width", None),
                "height": getattr(task.system_metadata, "height", None),
            }
        tasks.append({
            "task_id": task.id,
            "video_id": task.video_id,
            "status": task.status,
            "index_id": task.index_id,
            "system_metadata": meta,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
        })

    return {"tasks": tasks, "index_id": idx}


def list_indexed_videos(index_id=None, page=1, page_limit=10):
    client = _get_client()
    idx = index_id or TWELVELABS_INDEX_ID
    logger.info("Listing videos in index %s", idx)

    response = client.indexes.videos.list(
        index_id=idx,
        page=page,
        page_limit=page_limit,
        sort_by="created_at",
        sort_option="desc",
    )

    videos = []
    for video in response:
        meta = {}
        if video.system_metadata:
            meta = {
                "filename": getattr(video.system_metadata, "filename", None),
                "duration": getattr(video.system_metadata, "duration", None),
                "fps": getattr(video.system_metadata, "fps", None),
                "width": getattr(video.system_metadata, "width", None),
                "height": getattr(video.system_metadata, "height", None),
                "size": getattr(video.system_metadata, "size", None),
            }

        hls_url = None
        thumbnail_url = None
        if hasattr(video, "hls") and video.hls:
            hls_url = getattr(video.hls, "video_url", None)
            thumb_urls = getattr(video.hls, "thumbnail_urls", None)
            if thumb_urls:
                thumbnail_url = thumb_urls[0] if isinstance(thumb_urls, (list, tuple)) and len(thumb_urls) > 0 else (thumb_urls if isinstance(thumb_urls, str) else None)

        # List response often omits hls/thumbnails; enrich from get_video_info when missing
        if (not hls_url or not thumbnail_url) and video.id:
            try:
                info = get_video_info(video.id, index_id=idx)
                if info.get("hls"):
                    if not hls_url and info["hls"].get("video_url"):
                        hls_url = info["hls"]["video_url"]
                    if not thumbnail_url and info["hls"].get("thumbnail_urls"):
                        urls = info["hls"]["thumbnail_urls"]
                        thumbnail_url = urls[0] if isinstance(urls, (list, tuple)) and urls else (urls if isinstance(urls, str) else None)
            except Exception as e:
                logger.debug("Enrich video %s thumbnail: %s", video.id, e)

        videos.append({
            "video_id": video.id,
            "system_metadata": meta,
            "hls_url": hls_url,
            "thumbnail_url": thumbnail_url,
            "created_at": video.created_at,
            "updated_at": video.updated_at,
            "indexed_at": video.indexed_at,
        })

    return {"videos": videos, "index_id": idx}


def update_video_user_metadata(video_id, user_metadata, index_id=None):
    """Update user_metadata for a video (e.g. overview). Values must be string, int, float, or bool."""
    client = _get_client()
    idx = index_id or TWELVELABS_INDEX_ID
    logger.info("Updating user_metadata for video %s in index %s", video_id, idx)
    client.indexes.videos.update(
        index_id=idx,
        video_id=video_id,
        user_metadata=user_metadata,
    )


OVERVIEW_ABOUT_KEY = "overview_about"
OVERVIEW_TOPICS_KEY = "overview_topics"
OVERVIEW_CATEGORIES_KEY = "overview_categories"


def get_video_overview_from_user_metadata(user_metadata):
    """Extract overview (about, topics, categories) from TwelveLabs user_metadata."""
    if not user_metadata or not isinstance(user_metadata, dict):
        return None
    about = user_metadata.get(OVERVIEW_ABOUT_KEY)
    topics_raw = user_metadata.get(OVERVIEW_TOPICS_KEY)
    categories_raw = user_metadata.get(OVERVIEW_CATEGORIES_KEY)
    if about is None and not topics_raw and not categories_raw:
        return None
    try:
        topics = json.loads(topics_raw) if isinstance(topics_raw, str) else (topics_raw if isinstance(topics_raw, list) else [])
    except (json.JSONDecodeError, TypeError):
        topics = []
    try:
        categories = json.loads(categories_raw) if isinstance(categories_raw, str) else (categories_raw if isinstance(categories_raw, list) else [])
    except (json.JSONDecodeError, TypeError):
        categories = []
    return {
        "about": about if isinstance(about, str) else None,
        "topics": topics if isinstance(topics, list) else [],
        "categories": categories if isinstance(categories, list) else [],
    }


def set_video_overview(video_id, about=None, topics=None, categories=None, index_id=None):
    """Persist overview to TwelveLabs video user_metadata."""
    user_metadata = {}
    if about is not None and isinstance(about, str):
        user_metadata[OVERVIEW_ABOUT_KEY] = about
    if topics is not None:
        user_metadata[OVERVIEW_TOPICS_KEY] = json.dumps(topics) if isinstance(topics, list) else json.dumps([])
    if categories is not None:
        user_metadata[OVERVIEW_CATEGORIES_KEY] = json.dumps(categories) if isinstance(categories, list) else json.dumps([])
    if not user_metadata:
        return
    update_video_user_metadata(video_id, user_metadata, index_id=index_id)


def get_video_info(video_id, index_id=None):
    client = _get_client()
    idx = index_id or TWELVELABS_INDEX_ID
    logger.info("Retrieving video %s from index %s", video_id, idx)

    video = client.indexes.videos.retrieve(
        index_id=idx,
        video_id=video_id,
    )

    meta = {}
    if video.system_metadata:
        meta = {
            "filename": getattr(video.system_metadata, "filename", None),
            "duration": getattr(video.system_metadata, "duration", None),
            "fps": getattr(video.system_metadata, "fps", None),
            "width": getattr(video.system_metadata, "width", None),
            "height": getattr(video.system_metadata, "height", None),
            "size": getattr(video.system_metadata, "size", None),
        }

    hls_info = None
    if video.hls:
        hls_info = {
            "video_url": video.hls.video_url,
            "thumbnail_urls": video.hls.thumbnail_urls,
            "status": str(video.hls.status) if video.hls.status else None,
        }

    return {
        "video_id": video.id,
        "system_metadata": meta,
        "user_metadata": video.user_metadata,
        "hls": hls_info,
        "created_at": video.created_at,
        "updated_at": video.updated_at,
        "indexed_at": video.indexed_at,
    }


def delete_indexed_video(video_id, index_id=None):
    client = _get_client()
    idx = index_id or TWELVELABS_INDEX_ID
    logger.info("Deleting video %s from index %s", video_id, idx)
    client.indexes.videos.delete(index_id=idx, video_id=video_id)


def get_index_info(index_id=None):
    client = _get_client()
    idx = index_id or TWELVELABS_INDEX_ID
    logger.info("Retrieving index info for %s", idx)
    index = client.indexes.retrieve(index_id=idx)
    models = []
    if index.models:
        for m in index.models:
            models.append({
                "name": getattr(m, "name", None),
                "options": getattr(m, "options", None),
            })
    return {
        "index_id": index.id,
        "name": index.name,
        "models": models,
        "video_count": index.video_count,
        "total_duration": index.total_duration,
        "created_at": str(index.created_at) if index.created_at else None,
        "updated_at": str(index.updated_at) if index.updated_at else None,
    }


_entity_collection_id = TWELVELABS_ENTITY_COLLECTION_ID or None


def _ensure_entity_collection():
    global _entity_collection_id
    if _entity_collection_id:
        return _entity_collection_id

    client = _get_client()

    response = client.entity_collections.list(name=TWELVELABS_ENTITY_COLLECTION_NAME)
    for collection in response:
        if collection.name == TWELVELABS_ENTITY_COLLECTION_NAME:
            _entity_collection_id = collection.id
            logger.info("Found existing entity collection: %s", _entity_collection_id)
            return _entity_collection_id

    collection = client.entity_collections.create(
        name=TWELVELABS_ENTITY_COLLECTION_NAME,
        description="Face entities for video redaction pipeline",
    )
    _entity_collection_id = collection.id
    logger.info("Created entity collection: %s", _entity_collection_id)
    return _entity_collection_id


def get_entity_collection_id():
    return _ensure_entity_collection()


def list_entity_collections():
    client = _get_client()
    results = []
    response = client.entity_collections.list()
    for collection in response:
        results.append({
            "id": collection.id,
            "name": collection.name,
            "description": collection.description,
            "created_at": str(collection.created_at) if collection.created_at else None,
        })
    return results


def upload_face_asset(snap_path):
    client = _get_client()
    logger.info("Uploading face asset from %s", snap_path)
    asset = client.assets.create(
        method="direct",
        file=open(snap_path, "rb"),
    )
    logger.info("Created asset: %s", asset.id)
    return asset.id


def upload_face_asset_from_url(image_url):
    client = _get_client()
    logger.info("Uploading face asset from URL: %s", image_url)
    asset = client.assets.create(
        method="url",
        url=image_url,
    )
    logger.info("Created asset: %s", asset.id)
    return asset.id


def _serialize_entity(entity, metadata=None):
    entity_id = getattr(entity, "id", None)
    raw_metadata = metadata if metadata is not None else getattr(entity, "metadata", None)
    resolved_metadata = raw_metadata if isinstance(raw_metadata, dict) else None
    return {
        "id": entity_id,
        "entity_id": entity_id,
        "name": getattr(entity, "name", None),
        "description": getattr(entity, "description", None),
        "status": str(entity.status) if getattr(entity, "status", None) else None,
        "asset_ids": getattr(entity, "asset_ids", None),
        "metadata": resolved_metadata,
        "created_at": str(entity.created_at) if getattr(entity, "created_at", None) else None,
    }


def create_entity(name, asset_ids, description=None, metadata=None):
    client = _get_client()
    collection_id = _ensure_entity_collection()
    logger.info("Creating entity '%s' with %d assets in collection %s",
                name, len(asset_ids), collection_id)

    entity = client.entity_collections.entities.create(
        entity_collection_id=collection_id,
        name=name,
        asset_ids=asset_ids,
        **({"description": description} if description else {}),
        **({"metadata": metadata} if metadata else {}),
    )

    logger.info("Created entity: id=%s, status=%s", entity.id, entity.status)
    return _serialize_entity(entity, metadata=metadata)


def wait_for_entity_ready(entity_id, timeout=120):
    client = _get_client()
    collection_id = _ensure_entity_collection()
    start = time.time()

    while time.time() - start < timeout:
        entity = client.entity_collections.entities.retrieve(
            entity_collection_id=collection_id,
            entity_id=entity_id,
        )
        if entity.status and str(entity.status) == "ready":
            return True
        time.sleep(3)

    return False


def list_entities():
    client = _get_client()
    collection_id = _ensure_entity_collection()
    results = []
    response = client.entity_collections.entities.list(
        entity_collection_id=collection_id,
    )
    for entity in response:
        metadata = getattr(entity, "metadata", None)
        if metadata is None and getattr(entity, "id", None):
            try:
                detailed = client.entity_collections.entities.retrieve(
                    entity_collection_id=collection_id,
                    entity_id=entity.id,
                )
                metadata = getattr(detailed, "metadata", None)
            except Exception as e:
                logger.debug("Could not enrich entity %s metadata: %s", entity.id, e)
        results.append(_serialize_entity(entity, metadata=metadata))
    return results


def retrieve_entity(entity_id):
    client = _get_client()
    collection_id = _ensure_entity_collection()
    entity = client.entity_collections.entities.retrieve(
        entity_collection_id=collection_id,
        entity_id=entity_id,
    )
    return _serialize_entity(entity, metadata=getattr(entity, "metadata", None))


def delete_entity(entity_id):
    client = _get_client()
    collection_id = _ensure_entity_collection()
    client.entity_collections.entities.delete(
        entity_collection_id=collection_id,
        entity_id=entity_id,
    )
    logger.info("Deleted entity: %s", entity_id)


def entity_search(entity_id, query_suffix="", index_id=None):
    client = _get_client()
    idx = index_id or TWELVELABS_INDEX_ID

    query_text = f"<@{entity_id}>"
    if query_suffix:
        query_text = f"<@{entity_id}> {query_suffix}"

    logger.info("Entity search in index %s: %s", idx, query_text)

    response = client.search.query(
        index_id=idx,
        search_options=["visual"],
        query_text=query_text,
        group_by="video",
        sort_option="score",
        page_limit=50,
    )
    return _serialize_search_results(response)


def entity_search_time_ranges(entity_id, video_id=None, index_id=None):
    results = entity_search(entity_id, index_id=index_id)
    ranges = []
    for item in results:
        if "clips" in item:
            vid = item.get("video_id")
            if video_id and vid != video_id:
                continue
            for clip in item["clips"]:
                ranges.append({"start": clip["start"], "end": clip["end"]})
        else:
            vid = item.get("video_id")
            if video_id and vid != video_id:
                continue
            ranges.append({"start": item["start"], "end": item["end"]})
    return ranges


def create_entities_from_face_snaps(unique_faces, run_dir):
    collection_id = _ensure_entity_collection()
    entities = []
    import os

    for face in unique_faces:
        snap_path = face.get("snap_path")
        if not snap_path or not os.path.isfile(snap_path):
            continue

        person_id = face.get("person_id", "unknown")
        description = face.get("description", "")

        try:
            asset_id = upload_face_asset(snap_path)

            entity_result = create_entity(
                name=person_id,
                asset_ids=[asset_id],
                description=description or f"Detected face: {person_id}",
                metadata={"person_id": person_id, "source": "video_redaction_pipeline"},
            )

            face["entity_id"] = entity_result["entity_id"]
            face["entity_asset_ids"] = [asset_id]
            entities.append(entity_result)

            logger.info("Created entity for %s: entity_id=%s",
                        person_id, entity_result["entity_id"])
        except Exception as e:
            logger.warning("Failed to create entity for %s: %s", person_id, str(e))

    return entities
