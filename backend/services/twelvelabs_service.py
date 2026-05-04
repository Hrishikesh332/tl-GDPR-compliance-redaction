import json
import logging
import os
import time

from twelvelabs import TwelveLabs

from config import (
    TWELVELABS_API_KEY,
    TWELVELABS_ENTITY_COLLECTION_ID,
    TWELVELABS_ENTITY_COLLECTION_NAME,
    TWELVELABS_INDEX_ID,
)
from services.face_identity import get_face_identity
from services.twelvelabs_service_helpers import (
    ENTITY_PAGE_LIMIT,
    TWELVELABS_API_BASE,
    TWELVELABS_API_VERSION,
    dedupe_search_clips,
    extract_api_error,
    extract_list_items,
    extract_next_page_token,
    preferred_thumbnail_url,
    get_twelvelabs_api_url,
    get_value,
    merge_search_results,
    parse_json_markdown_response,
    raw_response_to_loggable,
    run_search_query as helper_run_search_query,
    search_query_supports_multi_media,
    serialize_entity,
    serialize_entity_collection,
    serialize_hls_info,
    serialize_search_results,
    serialize_task_system_metadata,
    serialize_video_system_metadata,
    list_twelvelabs_items as helper_list_twelvelabs_items,
    twelvelabs_api_request as helper_twelvelabs_api_request,
)

logger = logging.getLogger("video_redaction.twelvelabs")

twelvelabs_client = None
cached_entity_collection_id = TWELVELABS_ENTITY_COLLECTION_ID or None
TWELVELABS_SDK_TIMEOUT_SEC = float(os.environ.get("TWELVELABS_SDK_TIMEOUT_SEC", "180"))
TWELVELABS_ANALYZE_TIMEOUT_SEC = float(os.environ.get("TWELVELABS_ANALYZE_TIMEOUT_SEC", TWELVELABS_SDK_TIMEOUT_SEC))
TWELVELABS_REST_TIMEOUT_SEC = float(os.environ.get("TWELVELABS_REST_TIMEOUT_SEC", "180"))

PEOPLE_DESCRIPTION_PROMPT = (
    "List every distinct person visible in this video. "
    "Order the array so people whose faces should likely be anonymized or hidden "
    "for privacy come first. Put clearly official or public-facing people later "
    "(for example police officers in uniform, judges, anchors, presenters, or other "
    "on-duty public officials). "
    "For each person, provide: "
    "1) Their name if visible (from name tags, captions, chyrons, or on-screen text). "
    "   If the name is not identifiable, leave it blank. "
    "2) A brief physical description (hair, clothing, distinguishing features) "
    "3) The approximate time ranges (start and end in seconds) when they are visible. "
    "4) should_anonymize: true for private individuals or anyone whose face should be hidden; "
    "false for clearly official/public-facing people. "
    "5) is_official: true only when they are clearly an official/public-facing person. "
    "6) tags: include 'Anonymized' only when should_anonymize is true; do not include "
    "'Anonymized' for official/public-facing people. You may include 'Official' for those people. "
    "Return as a JSON array with objects having keys: "
    "name, description, time_ranges (array of {start_sec, end_sec}), "
    "should_anonymize, is_official, tags."
)

OBJECT_DESCRIPTION_PROMPT = (
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
)

SCENE_SUMMARY_PROMPT = (
    "Provide a detailed scene-by-scene summary of this video. "
    "Include scene changes, camera angles, lighting conditions, "
    "and any notable transitions. Return as JSON with key 'scenes' "
    "containing an array of {start_sec, end_sec, description}."
)

ANALYZE_FORMAT_INSTRUCTION = (
    "Format your response in clear markdown: use **bold** for emphasis, bullet or numbered lists "
    "where appropriate, and headings (##) for sections. When referencing specific moments in the "
    "video, include timestamps in mm:ss format (e.g. 02:30 or 1:05)."
)

OVERVIEW_ABOUT_KEY = "overview_about"
OVERVIEW_TOPICS_KEY = "overview_topics"
OVERVIEW_CATEGORIES_KEY = "overview_categories"

PIPELINE_METADATA_RESPONSE_FORMAT = {
    "type": "segment_definitions",
    "segment_definitions": [
        {
            "id": "face_redaction_target",
            "description": (
                "Return people segments for face redaction decisions. Create one segment per "
                "distinct face/person for each continuous time range where their face is visible "
                "enough to matter for redaction. Mark should_anonymize=true only when the face "
                "clearly needs privacy redaction. If the identity/category is ambiguous, the face "
                "is too small/blurred, or the person may be official/public-facing, set "
                "should_anonymize=false and review_required=true. Include official/public-facing "
                "people as non-redaction matches. Focus on faces and people, not objects, "
                "clothing-only shots, backs of heads, or crowd blobs where no face is visible."
            ),
            "fields": [
                {"name": "name", "type": "string", "description": "Visible name if known, otherwise blank."},
                {"name": "description", "type": "string", "description": "Face-focused visual match text: face visibility, hair/headwear, clothing, role, and distinguishing features."},
                {"name": "should_anonymize", "type": "boolean", "description": "True only when this visible face clearly should be redacted for privacy. Use false for official/public-facing people, ambiguous cases, low-confidence matches, unclear faces, or faces that need human review."},
                {"name": "is_official", "type": "boolean", "description": "True only for clearly official/public-facing people such as on-duty police, judges, anchors, presenters, or public officials; otherwise false."},
                {"name": "review_required", "type": "boolean", "description": "True when a human should verify whether this face should be redacted before applying blur."},
                {"name": "redaction_reason", "type": "string", "description": "Short reason for the recommendation, especially why auto-redaction is clear or why review is required."},
                {"name": "tags", "type": "string", "description": "Comma-separated tags. Include Anonymized only when should_anonymize is true. Include Official when is_official is true. Include Review when review_required is true."},
                {"name": "confidence", "type": "number", "description": "Confidence from 0 to 1."},
            ],
        },
        {
            "id": "scene_segment",
            "description": "Return concise scene-by-scene timeline context only. Do not list objects or create redaction targets here.",
            "fields": [
                {"name": "description", "type": "string", "description": "Scene summary, camera framing, setting, and notable transition."},
                {"name": "confidence", "type": "number", "description": "Confidence from 0 to 1."},
            ],
        },
    ],
}


def get_client():
    global twelvelabs_client
    if twelvelabs_client is None:
        twelvelabs_client = TwelveLabs(
            api_key=TWELVELABS_API_KEY,
            timeout=TWELVELABS_SDK_TIMEOUT_SEC,
        )
    return twelvelabs_client


def get_index_id():
    return TWELVELABS_INDEX_ID


def entity_api_available():
    return bool(TWELVELABS_API_KEY)


def entity_sdk_available():
    try:
        client = get_client()
        return hasattr(client, "entity_collections") and client.entity_collections is not None
    except Exception:
        return False


def resolve_index_id(index_id=None):
    return index_id or TWELVELABS_INDEX_ID


def parse_analysis_response(result_data, *, fallback_key, warning_message=None):
    parsed = parse_json_markdown_response(result_data)
    if parsed is not None:
        return parsed

    if warning_message:
        logger.warning(warning_message)
    return {fallback_key: result_data}


def serialize_task_summary(task):
    return {
        "task_id": task.id,
        "video_id": task.video_id,
        "status": task.status,
        "index_id": getattr(task, "index_id", None),
    }


def wait_for_task_completion(task_id, callback=None):
    client = get_client()

    def handle_task_update(task):
        logger.info("Task %s: status=%s", task_id, task.status)
        if callback:
            callback(task.status)

    return client.tasks.wait_for_done(
        task_id=task_id,
        sleep_interval=5.0,
        callback=handle_task_update,
    )


def analyze_video_to_json(video_id, prompt, *, fallback_key, warning_message, log_message):
    client = get_client()
    logger.info(log_message, video_id)
    result = client.analyze(
        video_id=video_id,
        prompt=prompt,
        temperature=0.1,
        request_options={"timeout_in_seconds": TWELVELABS_ANALYZE_TIMEOUT_SEC},
    )
    return parse_analysis_response(
        result.data,
        fallback_key=fallback_key,
        warning_message=warning_message,
    )


def ingest_video(video_path, callback=None):
    client = get_client()
    idx = get_index_id()
    logger.info("Uploading video to TwelveLabs index")

    with open(video_path, "rb") as video_file:
        task = client.tasks.create(
            index_id=idx,
            video_file=video_file,
            enable_video_stream=True,
        )

    logger.info("Created task %s, video_id=%s", task.id, task.video_id)
    completed = wait_for_task_completion(task.id, callback=callback)
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
    return serialize_task_summary(task)


def describe_people(video_id):
    return analyze_video_to_json(
        video_id,
        PEOPLE_DESCRIPTION_PROMPT,
        fallback_key="raw_description",
        warning_message="Could not parse people descriptions as JSON, returning raw text",
        log_message="Analyzing people in video %s",
    )


def describe_objects(video_id):
    return analyze_video_to_json(
        video_id,
        OBJECT_DESCRIPTION_PROMPT,
        fallback_key="raw_description",
        warning_message="Could not parse object descriptions as JSON, returning raw text",
        log_message="Analyzing objects in video %s",
    )


def get_scene_summary(video_id):
    return analyze_video_to_json(
        video_id,
        SCENE_SUMMARY_PROMPT,
        fallback_key="raw_summary",
        warning_message=None,
        log_message="Getting scene summary for video %s",
    )


def string_value(value):
    return value.strip() if isinstance(value, str) else ""


def as_float(value, default=None):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value == value else default
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def extract_pegasus_task_id(payload):
    if not isinstance(payload, dict):
        return ""
    for key in ("id", "_id", "task_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_pegasus_status(payload):
    if not isinstance(payload, dict):
        return "processing"
    raw = string_value(get_value(payload, "status", "state")).lower()
    if raw in {"ready", "completed", "complete", "succeeded", "success"}:
        return "ready"
    if raw in {"failed", "error"}:
        return "failed"
    if raw in {"queued", "pending"}:
        return "queued"
    return "processing"


def parse_json_text(value):
    if not isinstance(value, str):
        return None
    return parse_json_markdown_response(value)


def find_task_result(payload):
    if not isinstance(payload, dict):
        return payload
    for key in ("result", "results", "data", "output", "response"):
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return payload


def extract_segment_fields(segment):
    if not isinstance(segment, dict):
        return {}
    fields = {}
    for container_key in ("fields", "metadata", "attributes", "values", "response"):
        value = segment.get(container_key)
        if isinstance(value, dict):
            fields.update(value)
    fields.update(segment)
    return fields


def extract_segment_time(fields, *, start):
    keys = (
        ("start_sec", "start_time", "start", "begin", "from")
        if start
        else ("end_sec", "end_time", "end", "finish", "to")
    )
    value = as_float(get_value(fields, *keys), None)
    if value is not None:
        return max(0.0, value)
    nested = fields.get("time_range") or fields.get("timestamp") or fields.get("time")
    if isinstance(nested, dict):
        nested_value = as_float(get_value(nested, *keys), None)
        if nested_value is not None:
            return max(0.0, nested_value)
    return 0.0


def segments_from_payload(payload):
    if isinstance(payload, str):
        parsed = parse_json_text(payload)
        return segments_from_payload(parsed) if parsed is not None else []
    if isinstance(payload, list):
        return [(None, item) for item in payload]
    if not isinstance(payload, dict):
        return []

    segments = []
    definition_ids = {
        item.get("id")
        for item in PIPELINE_METADATA_RESPONSE_FORMAT.get("segment_definitions", [])
        if isinstance(item, dict)
    }
    for definition_id in definition_ids:
        value = payload.get(definition_id)
        if isinstance(value, list):
            segments.extend((definition_id, item) for item in value)
        elif isinstance(value, str):
            parsed = parse_json_text(value)
            if isinstance(parsed, list):
                segments.extend((definition_id, item) for item in parsed)

    if segments:
        return segments

    for key in ("segments", "items", "chapters", "highlights", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [(None, item) for item in value]
        if isinstance(value, str):
            parsed = parse_json_text(value)
            nested = segments_from_payload(parsed) if parsed is not None else []
            if nested:
                return nested

    result = payload.get("result")
    if result is not None and result is not payload:
        nested = segments_from_payload(result)
        if nested:
            return nested

    return []


def classify_pipeline_segment(definition_id, fields):
    raw = string_value(
        definition_id
        or get_value(fields, "segment_definition_id", "definition_id", "segment_id", "category", "type")
    ).lower()
    if "scene" in raw:
        return "scene"
    return "person"


def parse_tags(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def parse_pegasus_pipeline_metadata(task_payload):
    result = find_task_result(task_payload)
    segments = segments_from_payload(result)
    people_by_key = {}
    scenes = []

    for definition_id, segment in segments:
        fields = extract_segment_fields(segment)
        if not fields:
            continue
        start_sec = extract_segment_time(fields, start=True)
        end_sec = extract_segment_time(fields, start=False)
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        if end_sec == start_sec:
            end_sec = start_sec + 1.0

        kind = classify_pipeline_segment(definition_id, fields)
        if kind == "scene":
            description = string_value(get_value(fields, "description", "summary", "scene_description"))
            if description:
                scenes.append({
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "description": description,
                })
            continue

        name = string_value(get_value(fields, "name", "label", "person_name"))
        description = string_value(get_value(fields, "description", "summary", "visual_description"))
        if not name and not description:
            continue
        confidence = as_float(get_value(fields, "confidence", "score"), None)
        review_required = as_bool(get_value(fields, "review_required", "needs_review", "human_review"), False)
        redaction_reason = string_value(get_value(fields, "redaction_reason", "reason", "rationale"))
        key = (name.lower(), description.lower()[:120])
        entry = people_by_key.setdefault(key, {
            "name": name,
            "description": description,
            "time_ranges": [],
            "should_anonymize": as_bool(get_value(fields, "should_anonymize", "redact", "needs_redaction"), False),
            "is_official": as_bool(get_value(fields, "is_official", "official"), False),
            "review_required": review_required,
            "redaction_reason": redaction_reason,
            "tags": parse_tags(get_value(fields, "tags")),
        })
        if confidence is not None:
            entry["confidence"] = max(float(entry.get("confidence", 0.0) or 0.0), confidence)
        if not entry.get("name") and name:
            entry["name"] = name
        if len(description) > len(entry.get("description") or ""):
            entry["description"] = description
        entry["should_anonymize"] = bool(entry.get("should_anonymize")) or as_bool(
            get_value(fields, "should_anonymize", "redact", "needs_redaction"),
            False,
        )
        entry["is_official"] = bool(entry.get("is_official")) or as_bool(
            get_value(fields, "is_official", "official"),
            False,
        )
        entry["review_required"] = bool(entry.get("review_required")) or review_required
        if redaction_reason and len(redaction_reason) > len(entry.get("redaction_reason") or ""):
            entry["redaction_reason"] = redaction_reason
        tags = entry.setdefault("tags", [])
        for tag in parse_tags(get_value(fields, "tags")):
            if tag not in tags:
                tags.append(tag)
        entry["time_ranges"].append({
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
        })

    people = list(people_by_key.values())
    people.sort(key=lambda item: (
        not bool(item.get("should_anonymize")),
        item.get("time_ranges", [{}])[0].get("start_sec", 0.0),
        item.get("name") or item.get("description") or "",
    ))
    scenes.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    return {
        "people": people,
        "objects": [],
        "scene_summary": {"scenes": scenes},
    }


def create_pegasus_pipeline_metadata_task(asset_id):
    body = {
        "video": {
            "type": "asset_id",
            "asset_id": asset_id,
        },
        "model_name": "pegasus1.5",
        "analysis_mode": "time_based_metadata",
        "response_format": PIPELINE_METADATA_RESPONSE_FORMAT,
        "temperature": 0.1,
    }
    logger.info("Creating Pegasus 1.5 pipeline metadata task from asset id %s", asset_id)
    return twelvelabs_api_request(
        "POST",
        "analyze/tasks",
        json_body=body,
        expected_status=(200, 201, 202),
        timeout=TWELVELABS_REST_TIMEOUT_SEC,
    )


def retrieve_pegasus_pipeline_metadata_task(task_id):
    logger.info("Retrieving Pegasus pipeline metadata task %s", task_id)
    return twelvelabs_api_request(
        "GET",
        f"analyze/tasks/{task_id}",
        expected_status=(200,),
        timeout=TWELVELABS_REST_TIMEOUT_SEC,
    )


def wait_for_pegasus_pipeline_metadata(task_id, *, timeout_sec=None, poll_interval_sec=5.0):
    deadline = time.time() + float(timeout_sec or TWELVELABS_ANALYZE_TIMEOUT_SEC)
    while time.time() < deadline:
        payload = retrieve_pegasus_pipeline_metadata_task(task_id)
        status = extract_pegasus_status(payload if isinstance(payload, dict) else {})
        if status == "ready":
            return payload
        if status == "failed":
            raise RuntimeError("Pegasus 1.5 pipeline metadata task failed.")
        time.sleep(max(1.0, float(poll_interval_sec)))
    raise TimeoutError(
        f"Pegasus 1.5 pipeline metadata task did not finish within {int(timeout_sec or TWELVELABS_ANALYZE_TIMEOUT_SEC)} seconds."
    )


def describe_video_with_pegasus(video_id):
    task = create_pegasus_pipeline_metadata_task(video_id)
    task_id = extract_pegasus_task_id(task if isinstance(task, dict) else {})
    if not task_id:
        raise RuntimeError("TwelveLabs did not return a Pegasus 1.5 pipeline metadata task id.")

    status = extract_pegasus_status(task if isinstance(task, dict) else {})
    task_payload = task if status == "ready" else wait_for_pegasus_pipeline_metadata(task_id)
    metadata = parse_pegasus_pipeline_metadata(task_payload if isinstance(task_payload, dict) else {})
    metadata["task_id"] = task_id
    return metadata


def run_search_query(client, kwargs, *, image_path=None, image_paths=None, image_url=None):
    return helper_run_search_query(
        client,
        kwargs,
        image_path=image_path,
        image_paths=image_paths,
        image_url=image_url,
        logger=logger,
    )


def search_segments(query=None, index_id=None, image_paths=None, image_url=None, operator=None):
    client = get_client()
    idx = resolve_index_id(index_id)
    normalized_image_paths = [path for path in (image_paths or []) if path]
    query_input_count = int(bool(query)) + len(normalized_image_paths) + int(bool(image_url))
    normalized_operator = operator if operator in {"and", "or"} else ("and" if query_input_count > 1 else None)

    if not query and not normalized_image_paths and not image_url:
        raise ValueError("query text or image is required")

    logger.info(
        "Searching index (text=%s, image_count=%s, image_url=%s, operator=%s)",
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
            ranges.append({"start": item.start, "end": item.end})
    return ranges


def analyze_video_custom(video_id, prompt):
    client = get_client()
    logger.info("Custom analysis on video %s", video_id)
    enhanced_prompt = f"{ANALYZE_FORMAT_INSTRUCTION}\n\n{prompt}"
    result = client.analyze(
        video_id=video_id,
        prompt=enhanced_prompt,
        temperature=0.2,
        request_options={"timeout_in_seconds": TWELVELABS_ANALYZE_TIMEOUT_SEC},
    )
    return {"data": result.data, "id": result.id}


def index_video_from_file(video_path, index_id=None, enable_stream=True):
    client = get_client()
    idx = resolve_index_id(index_id)
    logger.info("Submitting indexing task for %s", video_path)

    with open(video_path, "rb") as video_file:
        task = client.tasks.create(
            index_id=idx,
            video_file=video_file,
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
    idx = resolve_index_id(index_id)
    logger.info("Submitting indexing task for URL")

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
    completed = wait_for_task_completion(task_id, callback=callback)
    return {
        "task_id": completed.id,
        "video_id": completed.video_id,
        "status": completed.status,
        "index_id": completed.index_id,
    }


def list_indexing_tasks(index_id=None, status_filter=None, page=1, page_limit=10):
    client = get_client()
    idx = resolve_index_id(index_id)

    kwargs = {
        "page": page,
        "page_limit": page_limit,
        "index_id": idx,
        "sort_by": "created_at",
        "sort_option": "desc",
    }
    if status_filter:
        kwargs["status"] = status_filter

    tasks = []
    for task in client.tasks.list(**kwargs):
        tasks.append(
            {
                "task_id": task.id,
                "video_id": task.video_id,
                "status": task.status,
                "index_id": task.index_id,
                "system_metadata": serialize_task_system_metadata(task.system_metadata),
                "created_at": task.created_at,
                "updated_at": task.updated_at,
            }
        )

    return {"tasks": tasks, "index_id": idx}


def list_indexed_videos(index_id=None, page=1, page_limit=10):
    client = get_client()
    idx = resolve_index_id(index_id)
    logger.info("Listing videos in index")

    response = client.indexes.videos.list(
        index_id=idx,
        page=page,
        page_limit=page_limit,
        sort_by="created_at",
        sort_option="desc",
    )

    videos = []
    for video in response:
        hls_url = None
        thumbnail_url = None
        raw_user_metadata = getattr(video, "user_metadata", None)
        if hasattr(video, "hls") and video.hls:
            hls_url = getattr(video.hls, "video_url", None)
            thumbnail_url = preferred_thumbnail_url(
                getattr(video.hls, "thumbnail_urls", None),
                user_metadata=raw_user_metadata,
            )

        if (not hls_url or not thumbnail_url or raw_user_metadata is None) and video.id:
            try:
                info = get_video_info(video.id, index_id=idx)
                if info.get("hls"):
                    if not hls_url:
                        hls_url = info["hls"].get("video_url")
                    enriched_user_metadata = info.get("user_metadata")
                    enriched_thumbnail = preferred_thumbnail_url(
                        info["hls"].get("thumbnail_urls"),
                        user_metadata=enriched_user_metadata,
                    )
                    if enriched_thumbnail:
                        thumbnail_url = enriched_thumbnail
            except Exception as exc:
                logger.debug("Enrich video %s thumbnail: %s", video.id, exc)

        videos.append(
            {
                "video_id": video.id,
                "system_metadata": serialize_video_system_metadata(video.system_metadata),
                "hls_url": hls_url,
                "thumbnail_url": thumbnail_url,
                "created_at": video.created_at,
                "updated_at": video.updated_at,
                "indexed_at": video.indexed_at,
            }
        )

    return {"videos": videos, "index_id": idx}


def update_video_user_metadata(video_id, user_metadata, index_id=None):
    """Update user_metadata for a video (e.g. overview)."""
    client = get_client()
    idx = resolve_index_id(index_id)
    logger.info("Updating user_metadata for video %s", video_id)
    client.indexes.videos.update(
        index_id=idx,
        video_id=video_id,
        user_metadata=user_metadata,
    )


def get_video_overview_from_user_metadata(user_metadata):
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
    user_metadata = {}
    if about is not None and isinstance(about, str):
        user_metadata[OVERVIEW_ABOUT_KEY] = about
    if topics is not None:
        user_metadata[OVERVIEW_TOPICS_KEY] = json.dumps(topics) if isinstance(topics, list) else json.dumps([])
    if categories is not None:
        user_metadata[OVERVIEW_CATEGORIES_KEY] = json.dumps(categories) if isinstance(categories, list) else json.dumps([])
    if user_metadata:
        update_video_user_metadata(video_id, user_metadata, index_id=index_id)


def get_video_info(video_id, index_id=None):
    client = get_client()
    idx = resolve_index_id(index_id)
    logger.info("Retrieving video %s", video_id)

    video = client.indexes.videos.retrieve(index_id=idx, video_id=video_id)
    return {
        "video_id": video.id,
        "system_metadata": serialize_video_system_metadata(video.system_metadata),
        "user_metadata": video.user_metadata,
        "hls": serialize_hls_info(video.hls),
        "created_at": video.created_at,
        "updated_at": video.updated_at,
        "indexed_at": video.indexed_at,
    }


def delete_indexed_video(video_id, index_id=None):
    client = get_client()
    idx = resolve_index_id(index_id)
    logger.info("Deleting video %s", video_id)
    client.indexes.videos.delete(index_id=idx, video_id=video_id)


def get_index_info(index_id=None):
    client = get_client()
    idx = resolve_index_id(index_id)
    logger.info("Retrieving index info")
    index = client.indexes.retrieve(index_id=idx)

    models = []
    if index.models:
        for model in index.models:
            models.append(
                {
                    "name": getattr(model, "name", None),
                    "options": getattr(model, "options", None),
                }
            )

    return {
        "index_id": index.id,
        "name": index.name,
        "models": models,
        "video_count": index.video_count,
        "total_duration": index.total_duration,
        "created_at": str(index.created_at) if index.created_at else None,
        "updated_at": str(index.updated_at) if index.updated_at else None,
    }


def twelvelabs_api_request(
    method,
    path,
    *,
    params=None,
    json_body=None,
    data=None,
    files=None,
    expected_status=(200,),
    timeout=None,
):
    effective_timeout = TWELVELABS_REST_TIMEOUT_SEC if timeout is None else timeout
    return helper_twelvelabs_api_request(
        method,
        path,
        api_key=TWELVELABS_API_KEY,
        params=params,
        json_body=json_body,
        data=data,
        files=files,
        expected_status=expected_status,
        timeout=effective_timeout,
    )


def list_twelvelabs_items(path, *, params=None):
    return helper_list_twelvelabs_items(path, api_key=TWELVELABS_API_KEY, params=params)


def ensure_entity_collection():
    global cached_entity_collection_id
    if cached_entity_collection_id:
        return cached_entity_collection_id

    if entity_sdk_available():
        client = get_client()
        response = client.entity_collections.list(name=TWELVELABS_ENTITY_COLLECTION_NAME)
        for collection in response:
            if collection.name == TWELVELABS_ENTITY_COLLECTION_NAME:
                cached_entity_collection_id = collection.id
                logger.info("Found existing entity collection: %s", cached_entity_collection_id)
                return cached_entity_collection_id

        collection = client.entity_collections.create(
            name=TWELVELABS_ENTITY_COLLECTION_NAME,
            description="Face entities for video redaction pipeline",
        )
        cached_entity_collection_id = collection.id
        logger.info("Created entity collection via SDK: %s", cached_entity_collection_id)
        return cached_entity_collection_id

    for collection in list_twelvelabs_items("entity-collections"):
        if get_value(collection, "name") == TWELVELABS_ENTITY_COLLECTION_NAME:
            cached_entity_collection_id = get_value(collection, "_id", "id")
            logger.info("Found existing entity collection via REST: %s", cached_entity_collection_id)
            return cached_entity_collection_id

    collection = twelvelabs_api_request(
        "POST",
        "entity-collections",
        json_body={
            "name": TWELVELABS_ENTITY_COLLECTION_NAME,
            "description": "Face entities for video redaction pipeline",
        },
        expected_status=(200, 201),
    )
    cached_entity_collection_id = get_value(collection, "_id", "id")
    logger.info("Created entity collection via REST: %s", cached_entity_collection_id)
    return cached_entity_collection_id


def get_entity_collection_id():
    return ensure_entity_collection()


def list_entity_collections():
    if entity_sdk_available():
        client = get_client()
        return [serialize_entity_collection(collection) for collection in client.entity_collections.list()]

    return [serialize_entity_collection(collection) for collection in list_twelvelabs_items("entity-collections")]


def upload_face_asset(snap_path):
    logger.info("Uploading face asset from %s", snap_path)
    client = get_client()
    try:
        with open(snap_path, "rb") as snap_file:
            asset = client.assets.create(method="direct", file=snap_file)
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
        asset = client.assets.create(method="url", url=image_url)
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


def create_pegasus_privacy_task(asset_id, *, response_format):
    """Create a Pegasus 1.5 async structured-analysis task from an existing asset id."""
    body = {
        "video": {
            "type": "asset_id",
            "asset_id": asset_id,
        },
        "model_name": "pegasus1.5",
        "analysis_mode": "time_based_metadata",
        "response_format": response_format,
        "temperature": 0.1,
    }
    logger.info("Creating Pegasus 1.5 privacy task from asset id %s", asset_id)
    return twelvelabs_api_request(
        "POST",
        "analyze/tasks",
        json_body=body,
        expected_status=(200, 201, 202),
        timeout=TWELVELABS_REST_TIMEOUT_SEC,
    )


def retrieve_pegasus_privacy_task(task_id):
    logger.info("Retrieving Pegasus privacy task %s", task_id)
    return twelvelabs_api_request(
        "GET",
        f"analyze/tasks/{task_id}",
        expected_status=(200,),
        timeout=TWELVELABS_REST_TIMEOUT_SEC,
    )


def create_entity(name, asset_ids, description=None, metadata=None):
    collection_id = ensure_entity_collection()
    logger.info(
        "Creating entity '%s' with %d assets in collection %s",
        name,
        len(asset_ids),
        collection_id,
    )

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
        body = {"name": name, "asset_ids": asset_ids}
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
    ensure_entity_collection()
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
        response = client.entity_collections.entities.list(entity_collection_id=collection_id)
        for entity in response:
            metadata = getattr(entity, "metadata", None)
            if metadata is None and getattr(entity, "id", None):
                try:
                    detailed = client.entity_collections.entities.retrieve(
                        entity_collection_id=collection_id,
                        entity_id=entity.id,
                    )
                    metadata = getattr(detailed, "metadata", None)
                except Exception as exc:
                    logger.debug("Could not enrich entity %s metadata: %s", entity.id, exc)
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
    idx = resolve_index_id(index_id)

    query_text = f"<@{entity_id}>"
    if query_suffix:
        query_text = f"<@{entity_id}> {query_suffix}"

    logger.info("Entity search: %s", query_text)
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
            item_video_id = item.get("video_id")
            if video_id and item_video_id != video_id:
                continue
            for clip in item["clips"]:
                ranges.append({"start": clip["start"], "end": clip["end"]})
            continue

        item_video_id = item.get("video_id")
        if video_id and item_video_id != video_id:
            continue
        ranges.append({"start": item["start"], "end": item["end"]})
    return ranges


def create_entities_from_face_snaps(unique_faces, run_dir):
    del run_dir
    ensure_entity_collection()
    entities = []

    for face in unique_faces:
        snap_path = face.get("snap_path")
        if not snap_path or not os.path.isfile(snap_path):
            continue

        person_id = get_face_identity(face) or "unknown"
        entity_name = str(face.get("name") or person_id).strip() or person_id
        description = face.get("description", "")

        try:
            asset_id = upload_face_asset(snap_path)
            entity_result = create_entity(
                name=entity_name,
                asset_ids=[asset_id],
                description=description or f"Detected face: {entity_name}",
                metadata={"person_id": person_id, "source": "video_redaction_pipeline"},
            )
            face["entity_id"] = entity_result["entity_id"]
            face["entity_asset_ids"] = [asset_id]
            entities.append(entity_result)
            logger.info("Created entity for %s: entity_id=%s", person_id, entity_result["entity_id"])
        except Exception as exc:
            logger.warning("Failed to create entity for %s: %s", person_id, str(exc))

    return entities
