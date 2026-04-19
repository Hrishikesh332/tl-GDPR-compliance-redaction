import inspect
import json
from contextlib import ExitStack

import requests

TWELVELABS_API_BASE = "https://api.twelvelabs.io"
TWELVELABS_API_VERSION = "v1.3"
ENTITY_PAGE_LIMIT = 50
PREFERRED_THUMBNAIL_URL_KEY = "preferred_thumbnail_url"


def parse_json_markdown_response(raw_text):
    if not isinstance(raw_text, str):
        return None

    text = raw_text.strip()
    if not text:
        return None

    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def raw_response_to_loggable(raw_list):
    """Convert raw API response items to JSON-serializable dicts for logging."""
    out = []
    for item in raw_list:
        clips = getattr(item, "clips", None)
        if clips is not None:
            out.append(
                {
                    "id": getattr(item, "id", None),
                    "score": getattr(item, "score", None),
                    "clips": [
                        {
                            "start": getattr(clip, "start", None),
                            "end": getattr(clip, "end", None),
                            "score": getattr(clip, "score", None),
                            "rank": getattr(clip, "rank", None),
                            "thumbnail_url": getattr(clip, "thumbnail_url", None),
                        }
                        for clip in clips
                    ],
                }
            )
        else:
            out.append(
                {
                    "video_id": getattr(item, "video_id", None),
                    "id": getattr(item, "id", None),
                    "start": getattr(item, "start", None),
                    "end": getattr(item, "end", None),
                    "score": getattr(item, "score", None),
                    "rank": getattr(item, "rank", None),
                    "thumbnail_url": getattr(item, "thumbnail_url", None),
                }
            )
    return out


def serialize_search_results(response):
    results_by_video = {}
    for item in response:
        grouped_clips = getattr(item, "clips", None)
        grouped_video_id = getattr(item, "id", None)
        video_id = getattr(item, "video_id", None) or grouped_video_id
        if not video_id:
            continue

        video_result = results_by_video.setdefault(
            video_id,
            {
                "video_id": video_id,
                "score": None,
                "clips": [],
            },
        )
        item_score = getattr(item, "score", None)
        if item_score is not None:
            current_score = video_result.get("score")
            if current_score is None or item_score > current_score:
                video_result["score"] = item_score

        if grouped_clips:
            for clip in grouped_clips:
                video_result["clips"].append(
                    {
                        "start": clip.start,
                        "end": clip.end,
                        "score": getattr(clip, "score", None),
                        "rank": getattr(clip, "rank", None),
                        "thumbnail_url": getattr(clip, "thumbnail_url", None),
                    }
                )
            continue

        start = getattr(item, "start", None)
        end = getattr(item, "end", None)
        if start is None or end is None:
            continue

        video_result["clips"].append(
            {
                "start": start,
                "end": end,
                "score": getattr(item, "score", None),
                "rank": getattr(item, "rank", None),
                "thumbnail_url": getattr(item, "thumbnail_url", None),
            }
        )

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
        result_maps.append(
            {
                item.get("video_id"): item
                for item in result_set or []
                if isinstance(item, dict) and item.get("video_id")
            }
        )

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
        merged.append(
            {
                "video_id": video_id,
                "score": max(scores) if scores else None,
                "clips": dedupe_search_clips(clips),
            }
        )

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


def log_search_results(raw_list, logger=None):
    if logger is None:
        return
    logger.info(
        "Search raw API response (%d items): %s",
        len(raw_list),
        json.dumps(raw_response_to_loggable(raw_list), default=str, indent=2),
    )


def run_search_query(
    client,
    kwargs,
    *,
    image_path=None,
    image_paths=None,
    image_url=None,
    logger=None,
):
    if image_url:
        response = client.search.query(
            **kwargs,
            query_media_type="image",
            query_media_url=image_url,
        )
        raw_list = list(response)
        log_search_results(raw_list, logger=logger)
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
            log_search_results(raw_list, logger=logger)
            return serialize_search_results(iter(raw_list))

        with ExitStack() as stack:
            media_files = [stack.enter_context(open(path, "rb")) for path in image_paths]
            response = client.search.query(
                **kwargs,
                query_media_type="image",
                query_media_files=media_files,
            )
            raw_list = list(response)
        log_search_results(raw_list, logger=logger)
        return serialize_search_results(iter(raw_list))

    response = client.search.query(**kwargs)
    raw_list = list(response)
    log_search_results(raw_list, logger=logger)
    return serialize_search_results(iter(raw_list))


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
    api_key,
    params=None,
    json_body=None,
    data=None,
    files=None,
    expected_status=(200,),
    timeout=60,
):
    if not api_key:
        raise RuntimeError("TWELVELABS_API_KEY is not configured.")

    response = requests.request(
        method=method,
        url=get_twelvelabs_api_url(path),
        headers={"x-api-key": api_key},
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


def list_twelvelabs_items(path, *, api_key, params=None):
    all_items = []
    next_page_token = None
    while True:
        query = dict(params or {})
        query.setdefault("page_limit", ENTITY_PAGE_LIMIT)
        if next_page_token:
            query["page_token"] = next_page_token
        payload = twelvelabs_api_request(
            "GET",
            path,
            api_key=api_key,
            params=query,
            expected_status=(200,),
        )
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


def serialize_task_system_metadata(system_metadata):
    if not system_metadata:
        return {}
    return {
        "filename": getattr(system_metadata, "filename", None),
        "duration": getattr(system_metadata, "duration", None),
        "width": getattr(system_metadata, "width", None),
        "height": getattr(system_metadata, "height", None),
    }


def serialize_video_system_metadata(system_metadata):
    if not system_metadata:
        return {}
    return {
        "filename": getattr(system_metadata, "filename", None),
        "duration": getattr(system_metadata, "duration", None),
        "fps": getattr(system_metadata, "fps", None),
        "width": getattr(system_metadata, "width", None),
        "height": getattr(system_metadata, "height", None),
        "size": getattr(system_metadata, "size", None),
    }


def metadata_preferred_thumbnail_url(user_metadata):
    if not isinstance(user_metadata, dict):
        return None
    value = user_metadata.get(PREFERRED_THUMBNAIL_URL_KEY)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def preferred_thumbnail_url(thumbnail_urls, user_metadata=None):
    metadata_url = metadata_preferred_thumbnail_url(user_metadata)
    if metadata_url:
        return metadata_url
    if isinstance(thumbnail_urls, (list, tuple)):
        if not thumbnail_urls:
            return None
        midpoint_index = len(thumbnail_urls) // 2
        return thumbnail_urls[midpoint_index]
    if isinstance(thumbnail_urls, str):
        return thumbnail_urls
    return None


def serialize_hls_info(hls):
    if not hls:
        return None
    return {
        "video_url": getattr(hls, "video_url", None),
        "thumbnail_urls": getattr(hls, "thumbnail_urls", None),
        "status": str(getattr(hls, "status", None)) if getattr(hls, "status", None) else None,
    }


def serialize_entity_collection(collection):
    created_at = get_value(collection, "created_at", "createdAt")
    return {
        "id": get_value(collection, "_id", "id"),
        "name": get_value(collection, "name"),
        "description": get_value(collection, "description"),
        "created_at": str(created_at) if created_at else None,
    }


def serialize_entity(entity, metadata=None):
    entity_id = get_value(entity, "_id", "id")
    raw_metadata = metadata if metadata is not None else get_value(entity, "metadata")
    resolved_metadata = raw_metadata if isinstance(raw_metadata, dict) else None
    created_at = get_value(entity, "created_at", "createdAt")
    status = get_value(entity, "status")
    return {
        "id": entity_id,
        "entity_id": entity_id,
        "name": get_value(entity, "name"),
        "description": get_value(entity, "description"),
        "status": str(status) if status else None,
        "asset_ids": get_value(entity, "asset_ids", "assetIds"),
        "metadata": resolved_metadata,
        "created_at": str(created_at) if created_at else None,
    }
