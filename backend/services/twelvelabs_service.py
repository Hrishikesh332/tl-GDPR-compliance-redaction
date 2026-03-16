import json
import inspect
import logging
import time
from contextlib import ExitStack

import requests
from twelvelabs import TwelveLabs

from config import (
    TWELVELABS_API_KEY, TWELVELABS_INDEX_ID,
    TWELVELABS_ENTITY_COLLECTION_ID, TWELVELABS_ENTITY_COLLECTION_NAME,
)

logger = logging.getLogger("video_redaction.twelvelabs")

_client = None
TWELVELABS_API_BASE = "https://api.twelvelabs.io"
TWELVELABS_API_VERSION = "v1.3"
# TwelveLabs entity collection/entity list endpoints currently cap page_limit at 50.
ENTITY_PAGE_LIMIT = 50


def get_client():
    global _client
    if _client is None:
        _client = TwelveLabs(api_key=TWELVELABS_API_KEY)
    return _client


def get_index_id():
    return TWELVELABS_INDEX_ID


def entity_api_available():
    return bool(TWELVELABS_API_KEY)


def ingest_video(video_path, callback=None):
    client = get_client()
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
    client = get_client()
    task = client.tasks.retrieve(task_id=task_id)
    return {
        "task_id": task.id,
        "video_id": task.video_id,
        "status": task.status,
        "index_id": task.index_id,
    }


def describe_people(video_id):
    client = get_client()
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
    client = get_client()
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
    client = get_client()
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


def raw_response_to_loggable(raw_list):
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


def serialize_search_results(response):
    results_by_video = {}
    for item in response:
        grouped_clips = getattr(item, "clips", None)
        grouped_video_id = getattr(item, "id", None)
        video_id = getattr(item, "video_id", None) or grouped_video_id
        if not video_id:
            continue

        video_result = results_by_video.setdefault(video_id, {
            "video_id": video_id,
            "score": None,
            "clips": [],
        })
        item_score = getattr(item, "score", None)
        if item_score is not None:
            current_score = video_result.get("score")
            if current_score is None or item_score > current_score:
                video_result["score"] = item_score

        if grouped_clips:
            for clip in grouped_clips:
                video_result["clips"].append({
                    "start": clip.start,
                    "end": clip.end,
                    "score": getattr(clip, "score", None),
                    "rank": getattr(clip, "rank", None),
                    "thumbnail_url": getattr(clip, "thumbnail_url", None),
                })
            continue

        start = getattr(item, "start", None)
        end = getattr(item, "end", None)
        if start is None or end is None:
            continue

        video_result["clips"].append({
            "start": getattr(item, "start", None),
            "end": getattr(item, "end", None),
            "score": getattr(item, "score", None),
            "rank": getattr(item, "rank", None),
            "thumbnail_url": getattr(item, "thumbnail_url", None),
        })

    results = list(results_by_video.values())
    for result in results:
        result["clips"].sort(
            key=lambda clip: (
                clip.get("rank") is None,
                clip.get("rank") if clip.get("rank") is not None else float("inf"),
                -(clip.get("score") or 0),
                clip.get("start") or 0,
            )
        )
    return results


def dedupe_search_clips(clips):
    deduped = {}
    for clip in clips or []:
        if not isinstance(clip, dict):
            continue
        key = (
            clip.get("start"),
            clip.get("end"),
            clip.get("rank"),
            clip.get("score"),
            clip.get("thumbnail_url"),
        )
        deduped.setdefault(key, clip)

    return sorted(
        deduped.values(),
        key=lambda clip: (
            clip.get("rank") is None,
            clip.get("rank") if clip.get("rank") is not None else float("inf"),
            -(clip.get("score") or 0),
            clip.get("start") or 0,
            clip.get("end") or 0,
        ),
    )


def merge_search_results(result_sets, operator="or"):
    normalized_operator = operator if operator in {"and", "or"} else "or"
    result_maps = []
    for result_set in result_sets:
        result_maps.append({
            item.get("video_id"): item
            for item in result_set or []
            if isinstance(item, dict) and item.get("video_id")
        })

    if not result_maps:
        return []

    if normalized_operator == "and":
        video_ids = set(result_maps[0].keys())
        for result_map in result_maps[1:]:
            video_ids &= set(result_map.keys())
    else:
        video_ids = set()
        for result_map in result_maps:
            video_ids |= set(result_map.keys())

    merged = []
    for video_id in video_ids:
        scores = []
        clips = []
        for result_map in result_maps:
            result = result_map.get(video_id)
            if not result:
                continue
            if result.get("score") is not None:
                scores.append(result["score"])
            clips.extend(result.get("clips") or [])
        merged.append({
            "video_id": video_id,
            "score": max(scores) if scores else None,
            "clips": dedupe_search_clips(clips),
        })

    return sorted(
        merged,
        key=lambda item: (
            -(item.get("score") or 0),
            item.get("video_id") or "",
        ),
    )


def search_query_supports_multi_media(client):
    try:
        params = inspect.signature(client.search.query).parameters
    except (TypeError, ValueError):
        return False
    return "query_media_files" in params


def run_search_query(client, kwargs, *, image_path=None, image_paths=None, image_url=None):
    if image_url:
        response = client.search.query(
            **kwargs,
            query_media_type="image",
            query_media_url=image_url,
        )
        raw_list = list(response)
        logger.info("Search raw API response (%d items): %s", len(raw_list), json.dumps(raw_response_to_loggable(raw_list), default=str, indent=2))
        return serialize_search_results(iter(raw_list))

    if image_paths:
        if len(image_paths) == 1:
            image_path = image_paths[0]

        if image_path:
            with open(image_path, "rb") as image_file:
                response = client.search.query(
                    **kwargs,
                    query_media_type="image",
                    query_media_file=image_file,
                )
            raw_list = list(response)
            logger.info("Search raw API response (%d items): %s", len(raw_list), json.dumps(raw_response_to_loggable(raw_list), default=str, indent=2))
            return serialize_search_results(iter(raw_list))

        with ExitStack() as stack:
            media_files = [stack.enter_context(open(path, "rb")) for path in image_paths]
            response = client.search.query(
                **kwargs,
                query_media_type="image",
                query_media_files=media_files,
            )
            raw_list = list(response)
        logger.info("Search raw API response (%d items): %s", len(raw_list), json.dumps(raw_response_to_loggable(raw_list), default=str, indent=2))
        return serialize_search_results(iter(raw_list))

    response = client.search.query(**kwargs)
    raw_list = list(response)
    logger.info("Search raw API response (%d items): %s", len(raw_list), json.dumps(raw_response_to_loggable(raw_list), default=str, indent=2))
    return serialize_search_results(iter(raw_list))


def search_segments(query=None, index_id=None, image_paths=None, image_url=None, operator=None):
    client = get_client()
    idx = index_id or TWELVELABS_INDEX_ID
    normalized_image_paths = [path for path in (image_paths or []) if path]
    query_input_count = int(bool(query)) + len(normalized_image_paths) + int(bool(image_url))
    normalized_operator = operator if operator in {"and", "or"} else ("and" if query_input_count > 1 else None)

    if not query and not normalized_image_paths and not image_url:
        raise ValueError("query text or image is required")

    logger.info(
        "Searching index %s (text=%s, image_count=%s, image_url=%s, operator=%s)",
        idx,
        bool(query),
        len(normalized_image_paths),
        bool(image_url),
        normalized_operator,
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
    if normalized_operator and query_input_count > 1:
        kwargs["operator"] = normalized_operator

    if image_url:
        return run_search_query(client, kwargs, image_url=image_url)

    if normalized_image_paths:
        if len(normalized_image_paths) == 1 or search_query_supports_multi_media(client):
            return run_search_query(client, kwargs, image_paths=normalized_image_paths)

        logger.warning(
            "Installed TwelveLabs SDK does not support query_media_files; falling back to per-image searches with operator=%s",
            normalized_operator or "or",
        )
        result_sets = [
            run_search_query(client, kwargs, image_path=image_path)
            for image_path in normalized_image_paths
        ]
        return merge_search_results(result_sets, operator=normalized_operator or "or")

    return run_search_query(client, kwargs)


def find_person_time_ranges(video_id, person_description):
    client = get_client()
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
    client = get_client()
    logger.info("Custom analysis on video %s", video_id)
    enhanced_prompt = f"{ANALYZE_FORMAT_INSTRUCTION}\n\n{prompt}"
    result = client.analyze(video_id=video_id, prompt=enhanced_prompt, temperature=0.2)
    return {"data": result.data, "id": result.id}


def index_video_from_file(video_path, index_id=None, enable_stream=True):
    client = get_client()
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
    client = get_client()
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
    client = get_client()

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
    client = get_client()
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
    client = get_client()
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
    client = get_client()
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
    client = get_client()
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
    client = get_client()
    idx = index_id or TWELVELABS_INDEX_ID
    logger.info("Deleting video %s from index %s", video_id, idx)
    client.indexes.videos.delete(index_id=idx, video_id=video_id)


def get_index_info(index_id=None):
    client = get_client()
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


def entity_sdk_available():
    try:
        client = get_client()
        return hasattr(client, "entity_collections") and client.entity_collections is not None
    except Exception:
        return False


def get_twelvelabs_api_url(path):
    return f"{TWELVELABS_API_BASE}/{TWELVELABS_API_VERSION}/{path.lstrip('/')}"


def extract_api_error(response):
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"TwelveLabs API request failed ({response.status_code})"

    if isinstance(payload, dict):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        inner_error = payload.get("error")
        if isinstance(inner_error, dict):
            for key in ("message", "detail", "error"):
                value = inner_error.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    return f"TwelveLabs API request failed ({response.status_code})"


def twelvelabs_api_request(
    method,
    path,
    *,
    params=None,
    json_body=None,
    data=None,
    files=None,
    expected_status=(200,),
    timeout=60,
):
    if not TWELVELABS_API_KEY:
        raise RuntimeError("TWELVELABS_API_KEY is not configured.")

    response = requests.request(
        method=method,
        url=get_twelvelabs_api_url(path),
        headers={"x-api-key": TWELVELABS_API_KEY},
        params=params,
        json=json_body,
        data=data,
        files=files,
        timeout=timeout,
    )
    if response.status_code not in expected_status:
        raise RuntimeError(extract_api_error(response))
    if response.status_code == 204 or not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def extract_list_items(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "entity_collections", "entities", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def extract_next_page_token(payload):
    if not isinstance(payload, dict):
        return None
    page_info = payload.get("page_info")
    if isinstance(page_info, dict):
        next_page_token = page_info.get("next_page_token")
        if isinstance(next_page_token, str) and next_page_token:
            return next_page_token
    next_page_token = payload.get("next_page_token")
    if isinstance(next_page_token, str) and next_page_token:
        return next_page_token
    return None


def list_twelvelabs_items(path, *, params=None):
    all_items = []
    next_page_token = None
    while True:
        query = dict(params or {})
        query.setdefault("page_limit", ENTITY_PAGE_LIMIT)
        if next_page_token:
            query["page_token"] = next_page_token
        payload = twelvelabs_api_request("GET", path, params=query, expected_status=(200,))
        all_items.extend(extract_list_items(payload))
        next_page_token = extract_next_page_token(payload)
        if not next_page_token:
            return all_items


def get_value(source, *keys):
    if isinstance(source, dict):
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
        return None
    for key in keys:
        value = getattr(source, key, None)
        if value is not None:
            return value
    return None


def serialize_entity_collection(collection):
    return {
        "id": get_value(collection, "_id", "id"),
        "name": get_value(collection, "name"),
        "description": get_value(collection, "description"),
        "created_at": str(get_value(collection, "created_at", "createdAt")) if get_value(collection, "created_at", "createdAt") else None,
    }


def ensure_entity_collection():
    global _entity_collection_id
    if _entity_collection_id:
        return _entity_collection_id

    if entity_sdk_available():
        client = get_client()
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
        logger.info("Created entity collection via SDK: %s", _entity_collection_id)
        return _entity_collection_id

    collections = list_twelvelabs_items("entity-collections")
    for collection in collections:
        if get_value(collection, "name") == TWELVELABS_ENTITY_COLLECTION_NAME:
            _entity_collection_id = get_value(collection, "_id", "id")
            logger.info("Found existing entity collection via REST: %s", _entity_collection_id)
            return _entity_collection_id

    collection = twelvelabs_api_request(
        "POST",
        "entity-collections",
        json_body={
            "name": TWELVELABS_ENTITY_COLLECTION_NAME,
            "description": "Face entities for video redaction pipeline",
        },
        expected_status=(200, 201),
    )
    _entity_collection_id = get_value(collection, "_id", "id")
    logger.info("Created entity collection via REST: %s", _entity_collection_id)
    return _entity_collection_id


def get_entity_collection_id():
    return ensure_entity_collection()


def list_entity_collections():
    if entity_sdk_available():
        client = get_client()
        results = []
        response = client.entity_collections.list()
        for collection in response:
            results.append(serialize_entity_collection(collection))
        return results

    return [serialize_entity_collection(collection) for collection in list_twelvelabs_items("entity-collections")]


def upload_face_asset(snap_path):
    logger.info("Uploading face asset from %s", snap_path)
    client = get_client()
    try:
        with open(snap_path, "rb") as snap_file:
            asset = client.assets.create(
                method="direct",
                file=snap_file,
            )
        logger.info("Created asset via SDK: %s", asset.id)
        return asset.id
    except AttributeError:
        with open(snap_path, "rb") as snap_file:
            asset = twelvelabs_api_request(
                "POST",
                "assets",
                data={"method": "direct"},
                files={"file": snap_file},
                expected_status=(200, 201),
                timeout=120,
            )
        asset_id = get_value(asset, "_id", "id")
        logger.info("Created asset via REST: %s", asset_id)
        return asset_id


def upload_face_asset_from_url(image_url):
    logger.info("Uploading face asset from URL: %s", image_url)
    client = get_client()
    try:
        asset = client.assets.create(
            method="url",
            url=image_url,
        )
        logger.info("Created asset via SDK: %s", asset.id)
        return asset.id
    except AttributeError:
        asset = twelvelabs_api_request(
            "POST",
            "assets",
            data={"method": "url", "url": image_url},
            expected_status=(200, 201),
        )
        asset_id = get_value(asset, "_id", "id")
        logger.info("Created asset via REST: %s", asset_id)
        return asset_id


def serialize_entity(entity, metadata=None):
    entity_id = get_value(entity, "_id", "id")
    raw_metadata = metadata if metadata is not None else get_value(entity, "metadata")
    resolved_metadata = raw_metadata if isinstance(raw_metadata, dict) else None
    return {
        "id": entity_id,
        "entity_id": entity_id,
        "name": get_value(entity, "name"),
        "description": get_value(entity, "description"),
        "status": str(get_value(entity, "status")) if get_value(entity, "status") else None,
        "asset_ids": get_value(entity, "asset_ids", "assetIds"),
        "metadata": resolved_metadata,
        "created_at": str(get_value(entity, "created_at", "createdAt")) if get_value(entity, "created_at", "createdAt") else None,
    }


def create_entity(name, asset_ids, description=None, metadata=None):
    collection_id = ensure_entity_collection()
    logger.info("Creating entity '%s' with %d assets in collection %s",
                name, len(asset_ids), collection_id)

    if entity_sdk_available():
        client = get_client()
        entity = client.entity_collections.entities.create(
            entity_collection_id=collection_id,
            name=name,
            asset_ids=asset_ids,
            **({"description": description} if description else {}),
            **({"metadata": metadata} if metadata else {}),
        )
    else:
        body = {
            "name": name,
            "asset_ids": asset_ids,
        }
        if description:
            body["description"] = description
        if metadata:
            body["metadata"] = metadata
        entity = twelvelabs_api_request(
            "POST",
            f"entity-collections/{collection_id}/entities",
            json_body=body,
            expected_status=(200, 201),
        )

    logger.info(
        "Created entity: id=%s, status=%s",
        get_value(entity, "_id", "id"),
        get_value(entity, "status"),
    )
    return serialize_entity(entity, metadata=metadata)


def wait_for_entity_ready(entity_id, timeout=120):
    collection_id = ensure_entity_collection()
    start = time.time()

    while time.time() - start < timeout:
        entity = retrieve_entity(entity_id)
        if entity.get("status") == "ready":
            return True
        time.sleep(3)

    return False


def list_entities():
    collection_id = ensure_entity_collection()
    if entity_sdk_available():
        client = get_client()
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
            results.append(serialize_entity(entity, metadata=metadata))
        return results

    entities = list_twelvelabs_items(f"entity-collections/{collection_id}/entities")
    return [serialize_entity(entity) for entity in entities]


def retrieve_entity(entity_id):
    collection_id = ensure_entity_collection()
    if entity_sdk_available():
        client = get_client()
        entity = client.entity_collections.entities.retrieve(
            entity_collection_id=collection_id,
            entity_id=entity_id,
        )
    else:
        entity = twelvelabs_api_request(
            "GET",
            f"entity-collections/{collection_id}/entities/{entity_id}",
            expected_status=(200,),
        )
    return serialize_entity(entity, metadata=get_value(entity, "metadata"))


def delete_entity(entity_id):
    collection_id = ensure_entity_collection()
    if entity_sdk_available():
        client = get_client()
        client.entity_collections.entities.delete(
            entity_collection_id=collection_id,
            entity_id=entity_id,
        )
    else:
        twelvelabs_api_request(
            "DELETE",
            f"entity-collections/{collection_id}/entities/{entity_id}",
            expected_status=(200, 204),
        )
    logger.info("Deleted entity: %s", entity_id)


def add_assets_to_entity(entity_id, asset_ids):
    if not asset_ids:
        raise ValueError("asset_ids is required")

    collection_id = ensure_entity_collection()
    logger.info(
        "Adding %d asset(s) to entity %s in collection %s",
        len(asset_ids),
        entity_id,
        collection_id,
    )

    if entity_sdk_available():
        client = get_client()
        entity = client.entity_collections.entities.create_assets(
            entity_collection_id=collection_id,
            entity_id=entity_id,
            asset_ids=asset_ids,
        )
    else:
        entity = twelvelabs_api_request(
            "POST",
            f"entity-collections/{collection_id}/entities/{entity_id}/assets",
            json_body={"asset_ids": asset_ids},
            expected_status=(200, 201),
        )

    return serialize_entity(entity, metadata=get_value(entity, "metadata"))


def entity_search(entity_id, query_suffix="", index_id=None):
    client = get_client()
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
    return serialize_search_results(response)


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
    collection_id = ensure_entity_collection()
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
