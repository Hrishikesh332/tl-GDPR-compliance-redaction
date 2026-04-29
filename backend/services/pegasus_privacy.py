import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from config import OUTPUT_DIR, SNAPS_DIR
from services import twelvelabs_service
from utils.storage import (
    get_run_dir,
    load_detection_metadata,
    load_job_manifest,
)

logger = logging.getLogger("video_redaction.pegasus_privacy")

PEGASUS_MODEL = "pegasus1.5"
PEGASUS_SCHEMA_VERSION = "1.2"
PEGASUS_PROMPT_VERSION = "privacy-assist-v4"
PEGASUS_ARTIFACT_FILENAME = "pegasus_privacy_assist.json"
PEGASUS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pegasus_assist")
PEGASUS_JOB_DIR = os.path.join(PEGASUS_OUTPUT_DIR, "jobs")
PEGASUS_ARTIFACT_DIR = os.path.join(PEGASUS_OUTPUT_DIR, "artifacts")

PRIVACY_CATEGORIES = {
    "person",
    "face",
    "screen",
    "document",
    "text",
    "license_plate",
    "logo",
    "object",
    "scene",
}
PRIVACY_SEVERITIES = {"low", "medium", "high"}
COURTROOM_EXCLUDED_PERSON_ROLES = {
    "judge",
    "lawyer",
    "attorney",
    "clerk",
    "officer",
    "police",
    "jury",
    "juror",
    "audience",
    "reporter",
    "bystander",
    "court_staff",
    "surrounding_participant",
    "public_official",
}
COURTROOM_ALLOWED_PERSON_TARGETS = {
    "main_verdict_subject",
    "protected_person",
    "private_non_party",
    "minor_or_vulnerable_person",
}

PEGASUS_PRIVACY_PROMPT = (
    "Return only segments that need redaction or careful human review. Do not return every "
    "visible person. For courtroom footage, explicitly ignore judges, lawyers, clerks, officers, "
    "jury members, audience members, reporters, and surrounding courtroom participants unless "
    "there is a clear privacy reason such as a protected witness, victim, minor, or private "
    "non-party. Focus person detections on the main verdict subject: the person whose verdict, "
    "sentencing, ruling, or legal outcome is being discussed. Include sensitive non-person "
    "targets only when redaction may be needed: IDs, documents, screens, names, addresses, "
    "phone numbers, license plates, and other PII. If a target needs care but exact geometry is "
    "uncertain, mark it as review-only or a custom-region prompt."
)

PEGASUS_RESPONSE_FORMAT = {
    "type": "segment_definitions",
    "segment_definitions": [
        {
            "id": "privacy_risk_segment",
            "description": (
                f"{PEGASUS_PRIVACY_PROMPT} Do not create broad background or crowd segments. Each "
                "segment must be narrow, actionable, and tied to one visible target that should be "
                "redacted or reviewed with care."
            ),
            "fields": [
                {
                    "name": "privacy_category",
                    "type": "string",
                    "description": (
                        "One of person, face, screen, document, text, license_plate, logo, object, scene. "
                        "Use scene only when the whole frame contains sensitive material; do not use it "
                        "for ordinary courtroom background."
                    ),
                    "enum": ["person", "face", "screen", "document", "text", "license_plate", "logo", "object", "scene"],
                },
                {
                    "name": "risk_level",
                    "type": "string",
                    "description": "One of low, medium, high.",
                    "enum": ["low", "medium", "high"],
                },
                {
                    "name": "label",
                    "type": "string",
                    "description": "Short target name, for example Main verdict subject, Protected witness, Visible ID, Phone screen, or License plate.",
                },
                {
                    "name": "description",
                    "type": "string",
                    "description": "What is visible and why this exact target needs redaction or careful review.",
                },
                {
                    "name": "reason",
                    "type": "string",
                    "description": (
                        "Specific reason this item should be redacted. For courtroom people, state why this is "
                        "the main verdict subject or another protected/private person; do not include generic "
                        "courtroom observers."
                    ),
                },
                {
                    "name": "redaction_target",
                    "type": "string",
                    "description": (
                        "One of main_verdict_subject, protected_person, private_non_party, minor_or_vulnerable_person, "
                        "sensitive_document, sensitive_screen, sensitive_identifier, license_plate, sensitive_object."
                    ),
                    "enum": [
                        "main_verdict_subject",
                        "protected_person",
                        "private_non_party",
                        "minor_or_vulnerable_person",
                        "sensitive_document",
                        "sensitive_screen",
                        "sensitive_identifier",
                        "license_plate",
                        "sensitive_object",
                    ],
                },
                {
                    "name": "scene_role",
                    "type": "string",
                    "description": (
                        "Role of the target in context. Use verdict_subject, defendant, respondent, or accused for "
                        "the main person whose verdict is being discussed. Ordinary judges, lawyers, clerks, officers, "
                        "jury, audience, reporters, and bystanders should not be segmented."
                    ),
                    "enum": [
                        "verdict_subject",
                        "defendant",
                        "respondent",
                        "accused",
                        "protected_witness",
                        "victim",
                        "minor",
                        "private_non_party",
                        "sensitive_item",
                        "unknown",
                    ],
                },
                {
                    "name": "redaction_decision",
                    "type": "string",
                    "description": (
                        "One of redact, handle_with_care, review_only. Do not emit a segment when the correct "
                        "decision is no redaction."
                    ),
                    "enum": ["redact", "handle_with_care", "review_only"],
                },
                {
                    "name": "subject_selection",
                    "type": "string",
                    "description": (
                        "Why this target is included. One of main_verdict_subject, protected_or_vulnerable_person, "
                        "sensitive_visual_detail, or not_applicable. Ordinary courtroom surrounding people are not applicable "
                        "and should not be emitted."
                    ),
                    "enum": [
                        "main_verdict_subject",
                        "protected_or_vulnerable_person",
                        "sensitive_visual_detail",
                        "not_applicable",
                    ],
                },
                {
                    "name": "inclusion_reason",
                    "type": "string",
                    "description": (
                        "A precise explanation of why this target qualifies for redaction review. For people, this must "
                        "say whether they are the main verdict subject or a protected/vulnerable person, not merely visible."
                    ),
                },
                {
                    "name": "handling_note",
                    "type": "string",
                    "description": "Short instruction for the editor, such as blur face, review timestamp, or draw custom region.",
                },
                {
                    "name": "recommended_action",
                    "type": "string",
                    "description": (
                        "One of select_detected_entity, select_object_class, create_review_bookmark, "
                        "jump_to_time, draw_custom_region_prompt."
                    ),
                    "enum": [
                        "select_detected_entity",
                        "select_object_class",
                        "create_review_bookmark",
                        "jump_to_time",
                        "draw_custom_region_prompt",
                    ],
                },
                {
                    "name": "object_class",
                    "type": "string",
                    "description": "Existing object class to blur when applicable, otherwise blank.",
                },
                {
                    "name": "entity_hint",
                    "type": "string",
                    "description": "Person/entity hint when applicable, otherwise blank.",
                },
                {
                    "name": "confidence",
                    "type": "number",
                    "description": "Confidence from 0 to 1.",
                },
            ],
        }
    ],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "video"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        logger.debug("Could not read Pegasus artifact at %s", path, exc_info=True)
        return None


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _get_value(source: Any, *keys: str) -> Any:
    if isinstance(source, dict):
        for key in keys:
            if key in source:
                return source[key]
    return None


def _string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value == value else default
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _normalize_severity(value: Any) -> str:
    text = _string_value(value).lower().replace(" ", "_")
    if text in PRIVACY_SEVERITIES:
        return text
    if text in {"critical", "severe"}:
        return "high"
    if text in {"moderate", "med"}:
        return "medium"
    return "medium"


def _normalize_category(value: Any) -> str:
    text = _string_value(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "license": "license_plate",
        "licenseplate": "license_plate",
        "plate": "license_plate",
        "pii": "text",
        "name": "text",
        "id": "document",
        "id_card": "document",
        "screen_text": "text",
        "person_face": "face",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in PRIVACY_CATEGORIES else "scene"


def _normalize_action_type(value: Any, category: str) -> str:
    text = _string_value(value).lower().replace("-", "_").replace(" ", "_")
    if text in {
        "select_detected_entity",
        "select_object_class",
        "create_review_bookmark",
        "jump_to_time",
        "draw_custom_region_prompt",
    }:
        return text
    if category in {"person", "face"}:
        return "select_detected_entity"
    if category in {"screen", "document", "text", "license_plate"}:
        return "draw_custom_region_prompt"
    if category in {"logo", "object"}:
        return "select_object_class"
    return "create_review_bookmark"


def _normalize_object_class(value: Any) -> str:
    return _string_value(value).lower()


def _structured_key(value: Any) -> str:
    return _string_value(value).lower().replace("-", "_").replace(" ", "_")


def _should_keep_privacy_target(
    *,
    category: str,
    label: str,
    description: str,
    reason: str,
    redaction_target: str,
    scene_role: str,
    redaction_decision: str,
    subject_selection: str,
    inclusion_reason: str,
) -> bool:
    decision_key = _structured_key(redaction_decision)
    if decision_key in {"no_redaction", "do_not_redact", "ignore", "exclude", "none"}:
        return False

    if category not in {"person", "face"}:
        return True

    target_key = _structured_key(redaction_target)
    role_key = _structured_key(scene_role)
    selection_key = _structured_key(subject_selection)
    text = " ".join([
        label,
        description,
        reason,
        inclusion_reason,
        redaction_target,
        scene_role,
        subject_selection,
    ]).lower()

    allowed_terms = (
        "main verdict subject",
        "verdict subject",
        "defendant",
        "respondent",
        "accused",
        "sentencing subject",
        "protected witness",
        "victim",
        "minor",
        "vulnerable",
        "private non-party",
        "private non party",
        "protected person",
    )
    has_allowed_reason = (
        target_key in COURTROOM_ALLOWED_PERSON_TARGETS
        or selection_key in {"main_verdict_subject", "protected_or_vulnerable_person"}
        or role_key in {"verdict_subject", "defendant", "respondent", "accused", "protected_witness", "victim", "minor", "private_non_party"}
        or any(term in text for term in allowed_terms)
    )
    if has_allowed_reason:
        return True

    if role_key in COURTROOM_EXCLUDED_PERSON_ROLES:
        return False

    excluded_terms = (
        "judge",
        "lawyer",
        "attorney",
        "clerk",
        "officer",
        "jury",
        "juror",
        "audience",
        "reporter",
        "bystander",
        "court staff",
        "surrounding courtroom",
        "courtroom participant",
    )
    generic_visibility_terms = (
        "person is visible",
        "visible person",
        "face is visible",
        "visible face",
        "appears in the courtroom",
        "shown in the courtroom",
    )
    if any(term in text for term in excluded_terms):
        return False
    if any(term in text for term in generic_visibility_terms):
        return False

    return True


def _extract_id(payload: dict[str, Any]) -> str:
    for key in ("id", "_id", "task_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_status(payload: dict[str, Any]) -> str:
    raw = _string_value(_get_value(payload, "status", "state")).lower()
    if raw in {"ready", "completed", "complete", "succeeded", "success"}:
        return "ready"
    if raw in {"failed", "error"}:
        return "failed"
    if raw in {"queued", "pending"}:
        return "queued"
    return "processing"


def _find_task_result(payload: dict[str, Any]) -> Any:
    for key in ("result", "results", "data", "output", "response"):
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return payload


def _find_segments(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, str):
        parsed = _parse_json_text(payload)
        return _find_segments(parsed) if parsed is not None else []
    if not isinstance(payload, dict):
        return []

    for key in ("segments", "items", "chapters", "highlights", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            parsed = _parse_json_text(value)
            nested = _find_segments(parsed) if parsed is not None else []
            if nested:
                return nested

    segment_definition_ids = {
        item.get("id") or item.get("name")
        for item in PEGASUS_RESPONSE_FORMAT.get("segment_definitions", [])
        if isinstance(item, dict)
    }
    grouped_segments: list[Any] = []
    for key, value in payload.items():
        if key in segment_definition_ids and isinstance(value, list):
            grouped_segments.extend(value)
    if grouped_segments:
        return grouped_segments

    for value in payload.values():
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            grouped_segments.extend(value)
    if grouped_segments:
        return grouped_segments

    result = payload.get("result")
    if result is not payload:
        nested = _find_segments(result)
        if nested:
            return nested

    return []


def _extract_segment_fields(segment: Any) -> dict[str, Any]:
    if not isinstance(segment, dict):
        return {}
    fields: dict[str, Any] = {}
    for container_key in ("fields", "metadata", "attributes", "values", "response"):
        value = segment.get(container_key)
        if isinstance(value, dict):
            fields.update(value)
    fields.update(segment)
    return fields


def _extract_time(fields: dict[str, Any], start: bool) -> float:
    keys = (
        ("start_sec", "start_time", "start", "begin", "from")
        if start
        else ("end_sec", "end_time", "end", "finish", "to")
    )
    value = _as_float(_get_value(fields, *keys), None)
    if value is not None:
        return max(0.0, value)
    nested = fields.get("time_range") or fields.get("timestamp") or fields.get("time")
    if isinstance(nested, dict):
        nested_value = _as_float(_get_value(nested, *keys), None)
        if nested_value is not None:
            return max(0.0, nested_value)
    return 0.0


def _task_result_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=True)
    except TypeError:
        return str(payload)


def _parse_json_text(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None


def _build_cache_metadata(
    *,
    video_id: str,
    local_job_id: str | None,
    source_fingerprint: str,
    cache_key: str,
    duration_sec: float,
    status: str,
    twelvelabs_task_id: str | None = None,
    source_type: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_id = f"pegasus_{_sha256_text(cache_key)[:16]}"
    return {
        "artifact_id": artifact_id,
        "job_id": artifact_id,
        "video_id": video_id,
        "local_job_id": local_job_id,
        "source_fingerprint": source_fingerprint,
        "cache_key": cache_key,
        "model": PEGASUS_MODEL,
        "prompt_version": PEGASUS_PROMPT_VERSION,
        "schema_version": PEGASUS_SCHEMA_VERSION,
        "duration_sec": duration_sec,
        "created_at": _utc_now_iso(),
        "status": status,
        "twelvelabs_task_id": twelvelabs_task_id,
        "source_type": source_type,
        "usage": usage or {},
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "overall_summary": "Pegasus has not returned privacy metadata for this video yet.",
        "privacy_risk_level": "low",
        "review_priority": "low",
    }


def _risk_from_events(events: list[dict[str, Any]]) -> str:
    if any(event.get("severity") == "high" for event in events):
        return "high"
    if any(event.get("severity") == "medium" for event in events):
        return "medium"
    return "low"


def _build_event_and_action(fields: dict[str, Any], index: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
    start_sec = _extract_time(fields, start=True)
    end_sec = _extract_time(fields, start=False)
    if end_sec < start_sec:
        end_sec = start_sec
    if end_sec == start_sec:
        end_sec = start_sec + 1.0

    category = _normalize_category(
        _get_value(fields, "privacy_category", "category", "type", "kind")
    )
    severity = _normalize_severity(
        _get_value(fields, "risk_level", "severity", "priority")
    )
    confidence = _as_float(_get_value(fields, "confidence", "score"), 0.65)
    confidence = max(0.0, min(1.0, confidence if confidence is not None else 0.65))
    label = _string_value(_get_value(fields, "label", "title", "name"))
    if not label:
        label = category.replace("_", " ").title()
    description = _string_value(_get_value(fields, "description", "summary"))
    reason = _string_value(_get_value(fields, "reason", "rationale", "privacy_reason"))
    inclusion_reason = _string_value(_get_value(fields, "inclusion_reason", "selection_reason", "target_reason"))
    if not reason:
        reason = inclusion_reason or description or f"Review {label} for privacy risk."
    redaction_target = _string_value(_get_value(fields, "redaction_target", "target_type", "privacy_target"))
    scene_role = _string_value(_get_value(fields, "scene_role", "role", "subject_role"))
    redaction_decision = _string_value(_get_value(fields, "redaction_decision", "decision", "privacy_decision"))
    subject_selection = _string_value(_get_value(fields, "subject_selection", "selection", "selection_type"))
    handling_note = _string_value(_get_value(fields, "handling_note", "handling", "review_note"))
    if not _should_keep_privacy_target(
        category=category,
        label=label,
        description=description,
        reason=reason,
        redaction_target=redaction_target,
        scene_role=scene_role,
        redaction_decision=redaction_decision,
        subject_selection=subject_selection,
        inclusion_reason=inclusion_reason,
    ):
        return None

    event_id = f"evt_{index + 1:03d}"
    action_id = f"act_{index + 1:03d}"
    action_type = _normalize_action_type(
        _get_value(fields, "recommended_action", "action", "action_type"),
        category,
    )
    object_class = _normalize_object_class(_get_value(fields, "object_class", "class", "object_type"))
    entity_hint = _string_value(_get_value(fields, "entity_hint", "person_hint", "entity_id", "person_id"))
    apply_mode = "automatic_if_matched" if action_type in {"select_detected_entity", "select_object_class"} else "review_only"

    event = {
        "id": event_id,
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "severity": severity,
        "category": category,
        "label": label[:120],
        "description": description[:600],
        "reason": reason[:600],
        "redaction_target": redaction_target[:120] or None,
        "scene_role": scene_role[:120] or None,
        "redaction_decision": redaction_decision[:120] or None,
        "subject_selection": subject_selection[:120] or None,
        "inclusion_reason": inclusion_reason[:600] or None,
        "handling_note": handling_note[:220] or None,
        "confidence": round(confidence, 3),
        "review_required": True,
        "recommended_action_ids": [action_id],
    }
    action = {
        "id": action_id,
        "type": action_type,
        "label": f"{label}: {action_type.replace('_', ' ')}",
        "reason": reason[:600],
        "confidence": round(confidence, 3),
        "event_ids": [event_id],
        "target": {
            "object_class": object_class or None,
            "entity_id": entity_hint or None,
            "redaction_target": redaction_target or None,
            "scene_role": scene_role or None,
            "redaction_decision": redaction_decision or None,
            "subject_selection": subject_selection or None,
            "inclusion_reason": inclusion_reason or None,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
        },
        "apply_mode": apply_mode,
    }
    return event, action


def _normalized_text_key(value: Any) -> str:
    text = _string_value(value).lower()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _can_merge_event_action_pair(
    previous_event: dict[str, Any],
    previous_action: dict[str, Any],
    current_event: dict[str, Any],
    current_action: dict[str, Any],
) -> bool:
    previous_start = _as_float(previous_event.get("start_sec"), 0.0) or 0.0
    previous_end = _as_float(previous_event.get("end_sec"), previous_start) or previous_start
    current_start = _as_float(current_event.get("start_sec"), 0.0) or 0.0
    if current_start > previous_end + 2.0:
        return False

    if previous_event.get("category") != current_event.get("category"):
        return False
    if previous_event.get("severity") != current_event.get("severity"):
        return False
    if previous_action.get("type") != current_action.get("type"):
        return False

    previous_target = previous_action.get("target") if isinstance(previous_action.get("target"), dict) else {}
    current_target = current_action.get("target") if isinstance(current_action.get("target"), dict) else {}
    previous_object = _normalized_text_key(previous_target.get("object_class"))
    current_object = _normalized_text_key(current_target.get("object_class"))
    if previous_object != current_object:
        return False

    previous_entity = _normalized_text_key(previous_target.get("entity_id"))
    current_entity = _normalized_text_key(current_target.get("entity_id"))
    previous_label = _normalized_text_key(previous_event.get("label"))
    current_label = _normalized_text_key(current_event.get("label"))
    if previous_entity and current_entity:
        return previous_entity == current_entity
    return previous_label == current_label


def _merge_event_action_pair(
    previous_event: dict[str, Any],
    previous_action: dict[str, Any],
    current_event: dict[str, Any],
    current_action: dict[str, Any],
) -> None:
    previous_start = _as_float(previous_event.get("start_sec"), 0.0) or 0.0
    previous_end = _as_float(previous_event.get("end_sec"), previous_start) or previous_start
    current_start = _as_float(current_event.get("start_sec"), 0.0) or 0.0
    current_end = _as_float(current_event.get("end_sec"), current_start) or current_start
    merged_start = round(min(previous_start, current_start), 3)
    merged_end = round(max(previous_end, current_end), 3)
    merged_confidence = round(
        max(
            _as_float(previous_event.get("confidence"), 0.0) or 0.0,
            _as_float(current_event.get("confidence"), 0.0) or 0.0,
        ),
        3,
    )

    previous_event["start_sec"] = merged_start
    previous_event["end_sec"] = merged_end
    previous_event["confidence"] = merged_confidence
    if len(_string_value(current_event.get("description"))) > len(_string_value(previous_event.get("description"))):
        previous_event["description"] = current_event.get("description", "")
    if len(_string_value(current_event.get("reason"))) > len(_string_value(previous_event.get("reason"))):
        previous_event["reason"] = current_event.get("reason", "")

    previous_action["confidence"] = merged_confidence
    previous_action["reason"] = previous_event.get("reason", "")
    previous_target = previous_action.get("target")
    if isinstance(previous_target, dict):
        previous_target["start_sec"] = merged_start
        previous_target["end_sec"] = merged_end


def _action_merge_key(event: dict[str, Any], action: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    target = action.get("target") if isinstance(action.get("target"), dict) else {}
    action_type = _normalized_text_key(action.get("type"))
    object_class = _normalized_text_key(target.get("object_class"))
    entity_id = _normalized_text_key(target.get("entity_id"))
    redaction_target = _normalized_text_key(target.get("redaction_target"))
    scene_role = _normalized_text_key(target.get("scene_role"))
    label_key = _normalized_text_key(event.get("label"))
    return (
        action_type,
        object_class,
        entity_id,
        redaction_target,
        scene_role,
        label_key,
    )


def normalize_pegasus_result(
    task_payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    task_result = _find_task_result(task_payload)
    raw_output = None

    if isinstance(task_result, str):
        parsed = _parse_json_text(task_result)
        if parsed is not None:
            task_result = parsed
        else:
            raw_output = task_result

    segments = _find_segments(task_result)
    merged_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, segment in enumerate(segments):
        fields = _extract_segment_fields(segment)
        if not fields:
            continue
        event_action = _build_event_and_action(fields, index)
        if event_action is None:
            continue
        event, action = event_action
        if merged_pairs:
            previous_event, previous_action = merged_pairs[-1]
            if _can_merge_event_action_pair(previous_event, previous_action, event, action):
                _merge_event_action_pair(previous_event, previous_action, event, action)
                continue
        merged_pairs.append((event, action))

    events: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    action_by_key: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for index, (event, action) in enumerate(merged_pairs):
        event_id = f"evt_{index + 1:03d}"
        event["id"] = event_id
        merge_key = _action_merge_key(event, action)
        existing_action = action_by_key.get(merge_key)
        if existing_action is None:
            action_id = f"act_{len(actions) + 1:03d}"
            action["id"] = action_id
            action["event_ids"] = [event_id]
            actions.append(action)
            action_by_key[merge_key] = action
            event["recommended_action_ids"] = [action_id]
        else:
            action_id = str(existing_action.get("id"))
            event_ids = existing_action.get("event_ids")
            if isinstance(event_ids, list) and event_id not in event_ids:
                event_ids.append(event_id)
            existing_confidence = _as_float(existing_action.get("confidence"), 0.0) or 0.0
            current_confidence = _as_float(action.get("confidence"), 0.0) or 0.0
            if current_confidence > existing_confidence:
                existing_action["confidence"] = round(current_confidence, 3)
            if len(_string_value(action.get("reason"))) > len(_string_value(existing_action.get("reason"))):
                existing_action["reason"] = action.get("reason", "")
            existing_target = existing_action.get("target")
            current_target = action.get("target")
            if isinstance(existing_target, dict) and isinstance(current_target, dict):
                existing_start = _as_float(existing_target.get("start_sec"), event.get("start_sec")) or 0.0
                existing_end = _as_float(existing_target.get("end_sec"), event.get("end_sec")) or existing_start
                current_start = _as_float(current_target.get("start_sec"), event.get("start_sec")) or 0.0
                current_end = _as_float(current_target.get("end_sec"), event.get("end_sec")) or current_start
                existing_target["start_sec"] = round(min(existing_start, current_start), 3)
                existing_target["end_sec"] = round(max(existing_end, current_end), 3)
            event["recommended_action_ids"] = [action_id]
        events.append(event)

    events.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["id"]))
    risk = _risk_from_events(events)
    summary_text = (
        f"Pegasus found {len(events)} privacy review segment"
        f"{'' if len(events) == 1 else 's'} for this video."
        if events
        else "Pegasus did not find obvious privacy hotspots in this video."
    )
    usage = _get_value(task_payload, "usage", "metadata") or metadata.get("usage") or {}
    next_metadata = {
        **metadata,
        "status": "ready",
        "usage": usage if isinstance(usage, dict) else {},
        "updated_at": _utc_now_iso(),
    }
    artifact = {
        "metadata": next_metadata,
        "summary": {
            "overall_summary": summary_text,
            "privacy_risk_level": risk,
            "review_priority": risk,
        },
        "timeline_events": events,
        "recommended_actions": actions,
    }
    if raw_output:
        artifact["raw_output"] = raw_output
    return artifact


def _resolve_video_info(video_id: str) -> dict[str, Any]:
    try:
        info = twelvelabs_service.get_video_info(video_id)
        return info if isinstance(info, dict) else {}
    except Exception as exc:
        logger.info("Could not retrieve TwelveLabs video info for Pegasus source: %s", exc)
        return {}


def _resolve_local_job(local_job_id: str | None, video_id: str) -> tuple[str | None, dict[str, Any]]:
    from services.pipeline import get_exact_job_id_by_video_id, get_job

    job_id = local_job_id or get_exact_job_id_by_video_id(video_id)
    if not job_id:
        return None, {}

    job = get_job(job_id) or load_job_manifest(job_id) or {}
    if not isinstance(job, dict):
        return None, {}
    if job.get("twelvelabs_video_id") and job.get("twelvelabs_video_id") != video_id:
        return None, {}
    return job_id, job


def _source_fingerprint(video_id: str, job: dict[str, Any], info: dict[str, Any]) -> tuple[str, float]:
    duration_sec = 0.0
    video_metadata = job.get("video_metadata") if isinstance(job.get("video_metadata"), dict) else {}
    if isinstance(video_metadata, dict):
        duration_sec = _as_float(video_metadata.get("duration"), 0.0) or _as_float(video_metadata.get("duration_sec"), 0.0) or 0.0

    video_path = _string_value(job.get("video_path"))
    if video_path and os.path.isfile(video_path):
        stat = os.stat(video_path)
        fingerprint = f"file:{os.path.basename(video_path)}:{stat.st_size}:{int(stat.st_mtime)}"
        return fingerprint, duration_sec

    system_metadata = info.get("system_metadata") if isinstance(info.get("system_metadata"), dict) else {}
    if isinstance(system_metadata, dict) and not duration_sec:
        duration_sec = _as_float(system_metadata.get("duration"), 0.0) or _as_float(system_metadata.get("duration_sec"), 0.0) or 0.0
    fingerprint = "|".join([
        f"video:{video_id}",
        f"updated:{info.get('updated_at') or ''}",
        f"indexed:{info.get('indexed_at') or ''}",
        f"filename:{system_metadata.get('filename') or ''}",
    ])
    return fingerprint, duration_sec


def _cache_key(video_id: str, source_fingerprint: str) -> str:
    return "|".join([
        video_id,
        source_fingerprint,
        PEGASUS_SCHEMA_VERSION,
        PEGASUS_PROMPT_VERSION,
    ])


def _artifact_path(video_id: str, local_job_id: str | None, cache_digest: str) -> str:
    if local_job_id:
        return os.path.join(get_run_dir(local_job_id), PEGASUS_ARTIFACT_FILENAME)
    return _canonical_artifact_path(video_id, cache_digest)


def _canonical_artifact_path(video_id: str, cache_digest: str) -> str:
    return os.path.join(PEGASUS_ARTIFACT_DIR, f"{_safe_id(video_id)}_{cache_digest[:12]}.json")


def _candidate_artifact_paths(video_id: str, local_job_id: str | None, cache_digest: str) -> list[str]:
    paths: list[str] = []
    if local_job_id:
        paths.append(os.path.join(get_run_dir(local_job_id), PEGASUS_ARTIFACT_FILENAME))
    paths.append(_canonical_artifact_path(video_id, cache_digest))
    if os.path.isdir(SNAPS_DIR):
        for run_id in sorted(os.listdir(SNAPS_DIR)):
            path = os.path.join(SNAPS_DIR, run_id, PEGASUS_ARTIFACT_FILENAME)
            if path not in paths:
                paths.append(path)
    return paths


def _job_path(job_id: str) -> str:
    return os.path.join(PEGASUS_JOB_DIR, f"{_safe_id(job_id)}.json")


def _sync_job_record(job_id: str, artifact: dict[str, Any], artifact_path: str, cached: bool) -> None:
    record = {
        "job_id": job_id,
        "status": artifact.get("metadata", {}).get("status", "processing"),
        "artifact_path": artifact_path,
        "cached": cached,
        "updated_at": _utc_now_iso(),
    }
    _write_json(_job_path(job_id), record)


def _load_record(job_id: str) -> dict[str, Any] | None:
    return _read_json(_job_path(job_id))


def _load_cached_artifact(path: str, cache_key: str) -> dict[str, Any] | None:
    artifact = _read_json(path)
    if not artifact:
        return None
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("cache_key") != cache_key:
        return None
    return artifact


def _load_video_artifact(path: str, video_id: str) -> dict[str, Any] | None:
    artifact = _read_json(path)
    if not artifact:
        return None
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("video_id") != video_id:
        return None
    if metadata.get("status") not in {None, "ready"}:
        return None
    return artifact


def _find_cached_artifact(
    video_id: str,
    local_job_id: str | None,
    cache_digest: str,
    cache_key: str,
) -> tuple[dict[str, Any], str] | tuple[None, None]:
    for path in _candidate_artifact_paths(video_id, local_job_id, cache_digest):
        artifact = _load_cached_artifact(path, cache_key)
        if artifact:
            return artifact, path
    return None, None


def _find_video_artifact(
    video_id: str,
    local_job_id: str | None,
    cache_digest: str,
) -> tuple[dict[str, Any], str] | tuple[None, None]:
    for path in _candidate_artifact_paths(video_id, local_job_id, cache_digest):
        artifact = _load_video_artifact(path, video_id)
        if artifact:
            return artifact, path
    return None, None


def get_cached_privacy_assist(video_id: str, *, local_job_id: str | None = None) -> dict[str, Any]:
    if not video_id:
        raise ValueError("video_id is required")

    resolved_job_id, job = _resolve_local_job(local_job_id, video_id)
    info = {} if _string_value(job.get("video_path")) else _resolve_video_info(video_id)
    source_fingerprint, _duration_sec = _source_fingerprint(video_id, job, info)
    cache_key = _cache_key(video_id, source_fingerprint)
    cache_digest = _sha256_text(cache_key)
    assist_job_id = f"pegasus_{cache_digest[:16]}"
    cached_artifact, path = _find_cached_artifact(video_id, resolved_job_id, cache_digest, cache_key)
    cache_status = "hit"
    if not cached_artifact or not path:
        cached_artifact, path = _find_video_artifact(video_id, resolved_job_id, cache_digest)
        cache_status = "video_id_fallback"
    if not cached_artifact or not path:
        raise FileNotFoundError("Cached Pegasus artifact not found")

    _sync_job_record(assist_job_id, cached_artifact, path, cached=True)
    status = cached_artifact.get("metadata", {}).get("status", "ready")
    return {
        "job_id": assist_job_id,
        "status": status,
        "cached": True,
        "cache_status": cache_status,
        "result": cached_artifact if status == "ready" else None,
    }


def start_privacy_assist_job(video_id: str, *, local_job_id: str | None = None, force: bool = False) -> dict[str, Any]:
    if not video_id:
        raise ValueError("video_id is required")

    resolved_job_id, job = _resolve_local_job(local_job_id, video_id)
    info = _resolve_video_info(video_id)
    source_fingerprint, duration_sec = _source_fingerprint(video_id, job, info)
    cache_key = _cache_key(video_id, source_fingerprint)
    cache_digest = _sha256_text(cache_key)
    assist_job_id = f"pegasus_{cache_digest[:16]}"
    path = _artifact_path(video_id, resolved_job_id, cache_digest)

    cached_artifact, cached_path = (None, None) if force else _find_cached_artifact(video_id, resolved_job_id, cache_digest, cache_key)
    if cached_artifact and cached_path:
        _sync_job_record(assist_job_id, cached_artifact, cached_path, cached=True)
        return {
            "job_id": assist_job_id,
            "status": cached_artifact.get("metadata", {}).get("status", "ready"),
            "cached": True,
            "result": cached_artifact if cached_artifact.get("metadata", {}).get("status") == "ready" else None,
        }

    task = twelvelabs_service.create_pegasus_privacy_task(
        video_id,
        response_format=PEGASUS_RESPONSE_FORMAT,
    )
    task_id = _extract_id(task if isinstance(task, dict) else {})
    if not task_id:
        raise RuntimeError("TwelveLabs did not return an analysis task id.")

    metadata = _build_cache_metadata(
        video_id=video_id,
        local_job_id=resolved_job_id,
        source_fingerprint=source_fingerprint,
        cache_key=cache_key,
        duration_sec=duration_sec,
        status=_extract_status(task if isinstance(task, dict) else {}),
        twelvelabs_task_id=task_id,
        source_type="asset_id_from_indexed_video_id",
    )
    artifact = {
        "metadata": metadata,
        "summary": _empty_summary(),
        "timeline_events": [],
        "recommended_actions": [],
    }
    _write_json(path, artifact)
    canonical_path = _canonical_artifact_path(video_id, cache_digest)
    if canonical_path != path:
        _write_json(canonical_path, artifact)
    _sync_job_record(assist_job_id, artifact, path, cached=False)
    return {
        "job_id": assist_job_id,
        "status": metadata["status"],
        "cached": False,
    }


def get_privacy_assist_job(job_id: str) -> dict[str, Any]:
    record = _load_record(job_id)
    if not record:
        raise FileNotFoundError("Pegasus job not found")

    artifact_path = _string_value(record.get("artifact_path"))
    artifact = _read_json(artifact_path) if artifact_path else None
    if not artifact:
        raise FileNotFoundError("Pegasus artifact not found")

    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    status = _string_value(metadata.get("status")) or _string_value(record.get("status")) or "processing"
    if status == "ready":
        return {"job_id": job_id, "status": "ready", "cached": bool(record.get("cached")), "result": artifact}
    if status == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "cached": bool(record.get("cached")),
            "error": artifact.get("error") or "Pegasus analysis failed.",
            "result": artifact,
        }

    task_id = _string_value(metadata.get("twelvelabs_task_id"))
    if not task_id:
        return {"job_id": job_id, "status": status, "cached": bool(record.get("cached"))}

    task_payload = twelvelabs_service.retrieve_pegasus_privacy_task(task_id)
    task_status = _extract_status(task_payload if isinstance(task_payload, dict) else {})
    if task_status == "ready":
        ready_artifact = normalize_pegasus_result(task_payload, metadata=metadata)
        _write_json(artifact_path, ready_artifact)
        cache_key = _string_value(metadata.get("cache_key"))
        video_id = _string_value(metadata.get("video_id"))
        if cache_key and video_id:
            canonical_path = _canonical_artifact_path(video_id, _sha256_text(cache_key))
            if canonical_path != artifact_path:
                _write_json(canonical_path, ready_artifact)
        _sync_job_record(job_id, ready_artifact, artifact_path, cached=bool(record.get("cached")))
        return {"job_id": job_id, "status": "ready", "cached": bool(record.get("cached")), "result": ready_artifact}

    if task_status == "failed":
        failed_artifact = {
            **artifact,
            "metadata": {**metadata, "status": "failed", "updated_at": _utc_now_iso()},
            "error": _string_value(_get_value(task_payload, "error", "message")) or "Pegasus analysis failed.",
        }
        _write_json(artifact_path, failed_artifact)
        _sync_job_record(job_id, failed_artifact, artifact_path, cached=bool(record.get("cached")))
        return {
            "job_id": job_id,
            "status": "failed",
            "cached": bool(record.get("cached")),
            "error": failed_artifact["error"],
            "result": failed_artifact,
        }

    artifact["metadata"] = {**metadata, "status": task_status, "updated_at": _utc_now_iso()}
    _write_json(artifact_path, artifact)
    _sync_job_record(job_id, artifact, artifact_path, cached=bool(record.get("cached")))
    return {"job_id": job_id, "status": task_status, "cached": bool(record.get("cached"))}


def _load_local_detection_index(local_job_id: str | None) -> dict[str, list[dict[str, Any]]]:
    if not local_job_id:
        return {"faces": [], "objects": []}
    metadata = load_detection_metadata(get_run_dir(local_job_id)) or {}
    faces = metadata.get("unique_faces")
    objects = metadata.get("unique_objects")
    return {
        "faces": faces if isinstance(faces, list) else [],
        "objects": objects if isinstance(objects, list) else [],
    }


def _face_matches(action: dict[str, Any], face: dict[str, Any]) -> bool:
    target = action.get("target") if isinstance(action.get("target"), dict) else {}
    entity_hint = _string_value(target.get("entity_id")).lower()
    label = _string_value(action.get("label")).lower()
    reason = _string_value(action.get("reason")).lower()
    candidates = [
        _string_value(face.get("entity_id")).lower(),
        _string_value(face.get("person_id")).lower(),
        _string_value(face.get("name")).lower(),
        _string_value(face.get("description")).lower(),
    ]
    if entity_hint and any(entity_hint == candidate for candidate in candidates if candidate):
        return True
    text = f"{label} {reason}"
    return any(candidate and candidate in text for candidate in candidates)


def _object_matches(action: dict[str, Any], obj: dict[str, Any]) -> bool:
    target = action.get("target") if isinstance(action.get("target"), dict) else {}
    target_class = _normalize_object_class(target.get("object_class"))
    object_class = _normalize_object_class(obj.get("identification") or obj.get("object_name") or obj.get("object_id"))
    if target_class and object_class and target_class == object_class:
        return True
    label = _string_value(action.get("label")).lower()
    reason = _string_value(action.get("reason")).lower()
    return bool(object_class and object_class in f"{label} {reason}")


def build_apply_preview(job_id: str, *, local_job_id: str | None = None) -> dict[str, Any]:
    job_response = get_privacy_assist_job(job_id)
    if job_response.get("status") != "ready":
        raise ValueError("Pegasus job is not ready yet.")

    artifact = job_response.get("result") if isinstance(job_response.get("result"), dict) else {}
    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    effective_local_job_id = local_job_id or metadata.get("local_job_id")
    detections = _load_local_detection_index(effective_local_job_id)
    faces = [item for item in detections["faces"] if isinstance(item, dict)]
    objects = [item for item in detections["objects"] if isinstance(item, dict)]

    can_apply: list[dict[str, Any]] = []
    review_only: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    selected_face_ids: set[str] = set()
    selected_object_classes: set[str] = set()
    review_event_ids: set[str] = set()

    actions = artifact.get("recommended_actions")
    if not isinstance(actions, list):
        actions = []

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = _string_value(action.get("type"))
        event_ids = [str(item) for item in action.get("event_ids", []) if item]

        if action_type == "select_detected_entity":
            matched_face = next((face for face in faces if _face_matches(action, face)), None)
            if matched_face:
                person_id = _string_value(matched_face.get("person_id"))
                if person_id:
                    selected_face_ids.add(person_id)
                    can_apply.append({
                        "action_id": action.get("id"),
                        "type": action_type,
                        "selection_id": f"face-{person_id}",
                        "person_id": person_id,
                        "label": matched_face.get("name") or matched_face.get("description") or person_id,
                        "event_ids": event_ids,
                        "reason": action.get("reason"),
                    })
                    continue
            review_only.append({
                "action_id": action.get("id"),
                "type": "create_review_bookmark",
                "label": action.get("label"),
                "event_ids": event_ids,
                "reason": action.get("reason"),
            })
            review_event_ids.update(event_ids)
            continue

        if action_type == "select_object_class":
            matched_object = next((obj for obj in objects if _object_matches(action, obj)), None)
            if matched_object:
                object_class = _normalize_object_class(matched_object.get("identification") or matched_object.get("object_name") or matched_object.get("object_id"))
                if object_class:
                    selected_object_classes.add(object_class)
                    can_apply.append({
                        "action_id": action.get("id"),
                        "type": action_type,
                        "selection_id": f"object-{object_class}",
                        "object_class": object_class,
                        "label": object_class,
                        "event_ids": event_ids,
                        "reason": action.get("reason"),
                    })
                    continue
            review_only.append({
                "action_id": action.get("id"),
                "type": "create_review_bookmark",
                "label": action.get("label"),
                "event_ids": event_ids,
                "reason": action.get("reason"),
            })
            review_event_ids.update(event_ids)
            continue

        if action_type in {"create_review_bookmark", "jump_to_time", "draw_custom_region_prompt"}:
            review_only.append({
                "action_id": action.get("id"),
                "type": action_type,
                "label": action.get("label"),
                "event_ids": event_ids,
                "reason": action.get("reason"),
            })
            review_event_ids.update(event_ids)
            continue

        unsupported.append({
            "action_id": action.get("id"),
            "type": action_type,
            "label": action.get("label"),
            "event_ids": event_ids,
            "reason": "This Pegasus recommendation cannot be mapped to an editor action yet.",
        })

    return {
        "can_apply": can_apply,
        "review_only": review_only,
        "unsupported": unsupported,
        "summary": {
            "selected_faces": len(selected_face_ids),
            "selected_object_classes": len(selected_object_classes),
            "review_bookmarks": len(review_event_ids),
        },
    }


def wait_briefly_for_ready(job_id: str, timeout_sec: float = 0.0) -> dict[str, Any]:
    deadline = time.time() + max(0.0, timeout_sec)
    response = get_privacy_assist_job(job_id)
    while response.get("status") not in {"ready", "failed"} and time.time() < deadline:
        time.sleep(1.0)
        response = get_privacy_assist_job(job_id)
    return response
