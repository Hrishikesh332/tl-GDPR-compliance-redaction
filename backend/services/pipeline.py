import logging
import os
import threading
import uuid
from datetime import datetime, timezone

import numpy as np

from config import (
    OUTPUT_DIR, KEYFRAME_INTERVAL_SEC, OBJECT_CONF_THRESHOLD,
    DEFAULT_BLUR_STRENGTH, DEFAULT_DETECT_EVERY_N,
)
from services import twelvelabs_service
from services.detection import detect_faces, detect_objects
from services.clustering import cluster_faces, cluster_objects
from services.redactor import redact_video
from utils.video import (
    extract_keyframes, extract_frames_at_timestamps,
    get_video_metadata, timestamps_from_time_ranges,
)
from utils.storage import get_run_dir, save_unique_face_snaps, save_unique_object_snaps, load_faces_objects_from_disk

logger = logging.getLogger("video_redaction.pipeline")

_jobs = {}
_lock = threading.Lock()


def _new_job_id():
    return str(uuid.uuid4())[:12]


def get_job(job_id):
    with _lock:
        return _jobs.get(job_id)


def get_job_id_by_video_id(video_id):
    """Return the first job_id whose twelvelabs_video_id matches, or None."""
    if not video_id:
        return None
    with _lock:
        for jid, j in _jobs.items():
            if j.get("twelvelabs_video_id") == video_id:
                return jid
    return None


def list_jobs():
    with _lock:
        return [
            {
                "job_id": jid,
                "status": j["status"],
                "created_at": j.get("created_at"),
                "video_filename": j.get("video_filename"),
            }
            for jid, j in _jobs.items()
        ]


def start_ingestion(video_path, video_filename=None, interval_sec=None, skip_indexing=False, existing_video_id=None):
    job_id = _new_job_id()
    interval = interval_sec or KEYFRAME_INTERVAL_SEC

    with _lock:
        _jobs[job_id] = {
            "status": "processing",
            "video_path": video_path,
            "video_filename": video_filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "twelvelabs_status": "pending",
            "local_status": "pending",
            "error": None,
        }

    thread = threading.Thread(
        target=_run_ingestion,
        args=(job_id, video_path, interval, skip_indexing, existing_video_id),
        daemon=True,
    )
    thread.start()
    return job_id


def _run_ingestion(job_id, video_path, interval_sec, skip_indexing=False, existing_video_id=None):
    try:
        logger.info("[Job %s] Pipeline started (interval=%.1fs, skip_indexing=%s)", job_id, interval_sec, skip_indexing)

        # ── STEP 1: Video metadata ──────────────────────────────────────
        logger.info("[Job %s] STEP 1/7: Getting video metadata...", job_id)
        metadata = get_video_metadata(video_path)
        with _lock:
            _jobs[job_id]["video_metadata"] = metadata
        logger.info("[Job %s] STEP 1/7: Done — %sx%s, %.1fs, %.0f fps", job_id,
                    metadata.get("width"), metadata.get("height"),
                    metadata.get("duration_sec"), metadata.get("fps"))

        # ── STEP 2: TwelveLabs upload & index (skipped if already indexed) ──
        video_id = (existing_video_id or "").strip() or None
        people_desc = []
        objects_desc = []
        scene_summary = {}

        if skip_indexing and video_id:
            logger.info("[Job %s] STEP 2/7: Skipping indexing — using existing video_id=%s", job_id, video_id)
            with _lock:
                _jobs[job_id]["twelvelabs_video_id"] = video_id
                _jobs[job_id]["twelvelabs_status"] = "indexed"
        elif skip_indexing:
            logger.info("[Job %s] STEP 2/7: Skipping indexing (no video_id provided)", job_id)
            with _lock:
                _jobs[job_id]["twelvelabs_status"] = "skipped"
        else:
            logger.info("[Job %s] STEP 2/7: TwelveLabs — uploading and indexing video...", job_id)
            with _lock:
                _jobs[job_id]["twelvelabs_status"] = "uploading"

            try:
                tl_result = twelvelabs_service.ingest_video(video_path)
                video_id = tl_result["video_id"]
                task_id = tl_result["task_id"]

                with _lock:
                    _jobs[job_id]["twelvelabs_video_id"] = video_id
                    _jobs[job_id]["twelvelabs_task_id"] = task_id
                    _jobs[job_id]["twelvelabs_status"] = "indexed"

                logger.info("[Job %s] STEP 2/7: Done — indexed, video_id=%s", job_id, video_id)

            except Exception as e:
                logger.warning("[Job %s] STEP 2/7: TwelveLabs indexing failed: %s (will use interval keyframes)", job_id, str(e))
                with _lock:
                    _jobs[job_id]["twelvelabs_status"] = f"index_failed: {str(e)}"

        # ── STEP 3: TwelveLabs analysis → get timestamps ────────────────
        if video_id:
            logger.info("[Job %s] STEP 3/7: TwelveLabs — analyzing (people, objects, scenes)...", job_id)
            with _lock:
                _jobs[job_id]["twelvelabs_status"] = "analyzing"

            try:
                people_desc = twelvelabs_service.describe_people(video_id)
                objects_desc = twelvelabs_service.describe_objects(video_id)
                scene_summary = twelvelabs_service.get_scene_summary(video_id)

                with _lock:
                    _jobs[job_id]["twelvelabs_people"] = people_desc
                    _jobs[job_id]["twelvelabs_objects"] = objects_desc
                    _jobs[job_id]["twelvelabs_scene_summary"] = scene_summary
                    _jobs[job_id]["twelvelabs_status"] = "analyzed"

                logger.info("[Job %s] STEP 3/7: Done — analysis complete", job_id)

            except Exception as e:
                logger.warning("[Job %s] STEP 3/7: TwelveLabs analysis failed: %s", job_id, str(e))
                with _lock:
                    _jobs[job_id]["twelvelabs_status"] = f"analysis_failed: {str(e)}"
        else:
            logger.info("[Job %s] STEP 3/7: Skipped (no video_id)", job_id)

        # ── STEP 4: Extract keyframes ───────────────────────────────────
        # Prefer timestamps from TwelveLabs analysis; fall back to
        # uniform interval sampling when analysis didn't produce ranges.
        logger.info("[Job %s] STEP 4/7: Extracting keyframes...", job_id)
        with _lock:
            _jobs[job_id]["local_status"] = "extracting_keyframes"

        analyzed_timestamps = _collect_timestamps_from_analysis(
            people_desc, objects_desc
        )

        if analyzed_timestamps:
            keyframes = extract_frames_at_timestamps(video_path, analyzed_timestamps)
            logger.info("[Job %s] STEP 4/7: Done — %d keyframes from analysis timestamps", job_id, len(keyframes))
        else:
            keyframes = extract_keyframes(video_path, interval_sec=interval_sec)
            logger.info("[Job %s] STEP 4/7: Done — %d keyframes at %.1fs interval (no analysis timestamps)", job_id, len(keyframes), interval_sec)

        # ── STEP 5: Face + object detection on keyframes ────────────────
        logger.info("[Job %s] STEP 5/7: Face and object detection on %d keyframes...", job_id, len(keyframes))
        with _lock:
            _jobs[job_id]["local_status"] = "detecting"

        all_faces = []
        all_objects = []

        for kf in keyframes:
            faces = detect_faces(kf["frame"], with_encodings=True)
            objects = detect_objects(kf["frame"], conf_threshold=OBJECT_CONF_THRESHOLD)
            for f in faces:
                f["frame_idx"] = kf["frame_idx"]
                f["timestamp"] = kf["timestamp"]
                all_faces.append(f)
            for o in objects:
                o["frame_idx"] = kf["frame_idx"]
                o["timestamp"] = kf["timestamp"]
                all_objects.append(o)

        logger.info("[Job %s] STEP 5/7: Done — %d face detections, %d object detections", job_id, len(all_faces), len(all_objects))

        # ── STEP 6: Clustering ──────────────────────────────────────────
        logger.info("[Job %s] STEP 6/7: Clustering into unique faces and objects...", job_id)
        with _lock:
            _jobs[job_id]["local_status"] = "clustering"

        unique_faces = cluster_faces(all_faces)
        unique_objects = cluster_objects(all_objects)
        logger.info("[Job %s] STEP 6/7: Done — %d unique faces, %d unique objects", job_id, len(unique_faces), len(unique_objects))

        # ── STEP 7: Save snapshots ──────────────────────────────────────
        logger.info("[Job %s] STEP 7/7: Saving snapshots to disk...", job_id)
        run_dir = get_run_dir(job_id)
        save_unique_face_snaps(run_dir, unique_faces)
        save_unique_object_snaps(run_dir, unique_objects)

        _enrich_faces_with_descriptions(job_id, people_desc, unique_faces)

        with _lock:
            _jobs[job_id]["local_status"] = "done"
            _jobs[job_id]["unique_faces"] = unique_faces
            _jobs[job_id]["unique_objects"] = unique_objects
            _jobs[job_id]["total_face_detections"] = len(all_faces)
            _jobs[job_id]["total_object_detections"] = len(all_objects)
            _jobs[job_id]["status"] = "ready"
            if video_id and "failed" not in _jobs[job_id].get("twelvelabs_status", ""):
                _jobs[job_id]["twelvelabs_status"] = "done"
        logger.info("[Job %s] STEP 7/7: Done — snapshots saved", job_id)

        logger.info("[Job %s] Pipeline complete — status=ready", job_id)

    except Exception as e:
        logger.error("[Job %s] Pipeline failed: %s", job_id, str(e), exc_info=True)
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)


def _collect_timestamps_from_analysis(people_desc, objects_desc):
    """Extract smart keyframe timestamps from TwelveLabs analysis time ranges.

    Only collects the actual time ranges where people/objects were detected,
    merges overlapping ranges, and samples sparsely (every ~5s) within them.
    """
    all_ranges = []

    if isinstance(people_desc, list):
        for entry in people_desc:
            if isinstance(entry, dict):
                ranges = entry.get("time_ranges", [])
                for tr in ranges:
                    all_ranges.append(tr)
                name = entry.get("name") or entry.get("description", "")[:40]
                if ranges:
                    logger.info("  People range: '%s' -> %d intervals", name, len(ranges))

    if isinstance(objects_desc, list):
        for entry in objects_desc:
            if isinstance(entry, dict):
                ranges = entry.get("time_ranges", [])
                for tr in ranges:
                    all_ranges.append(tr)
                name = entry.get("object_name") or entry.get("identification", "")[:40]
                if ranges:
                    logger.info("  Object range: '%s' -> %d intervals", name, len(ranges))

    if not all_ranges:
        return []

    logger.info("Total raw time ranges from analysis: %d", len(all_ranges))
    return timestamps_from_time_ranges(all_ranges)


def _enrich_faces_with_descriptions(job_id, people_desc, unique_faces=None):
    faces = unique_faces
    if faces is None:
        with _lock:
            job = _jobs.get(job_id)
            if not job or not job.get("unique_faces"):
                return
            faces = job["unique_faces"]

    if not faces:
        return

    if isinstance(people_desc, list):
        for i, face in enumerate(faces):
            if i < len(people_desc):
                desc_entry = people_desc[i]
                if isinstance(desc_entry, dict):
                    face["description"] = desc_entry.get("description", "")
                    face["time_ranges"] = desc_entry.get("time_ranges", [])
                    raw_name = desc_entry.get("name") or desc_entry.get("person_name") or ""
                    if raw_name:
                        face["name"] = raw_name
                        face["person_id"] = raw_name

    for i, face in enumerate(faces):
        if not face.get("name"):
            face["name"] = face.get("person_id", f"person_{i}")


def push_job_entities_to_twelvelabs(job_id):
    """
    Push this job's face snaps to TwelveLabs as entities.
    Only call this when the user explicitly requests it via the API.
    Returns list of created entity dicts; updates job['entities'] and face['entity_id'] in place.
    """
    job = get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    if job["status"] not in ("ready",):
        raise ValueError(f"Job {job_id} is not ready (status={job['status']})")
    unique_faces = job.get("unique_faces", [])
    if not unique_faces:
        return []
    run_dir = get_run_dir(job_id)
    entities = twelvelabs_service.create_entities_from_face_snaps(unique_faces, run_dir)
    with _lock:
        _jobs[job_id]["entities"] = entities
    logger.info("Job %s: pushed %d entities to TwelveLabs (user-requested)", job_id, len(entities))
    return entities


def get_enriched_faces(job_id):
    job = get_job(job_id)
    if job:
        return {
            "status": job["status"],
            "unique_faces": job.get("unique_faces", []),
            "unique_objects": job.get("unique_objects", []),
            "entities": job.get("entities", []),
            "twelvelabs_people": job.get("twelvelabs_people"),
            "twelvelabs_objects": job.get("twelvelabs_objects"),
            "twelvelabs_scene_summary": job.get("twelvelabs_scene_summary"),
            "video_metadata": job.get("video_metadata"),
            "total_face_detections": job.get("total_face_detections", 0),
            "total_object_detections": job.get("total_object_detections", 0),
        }
    # Fallback: load from disk when job is not in memory (e.g. after server restart)
    disk = load_faces_objects_from_disk(job_id)
    if disk:
        return {
            "status": "ready",
            "unique_faces": disk["unique_faces"],
            "unique_objects": disk["unique_objects"],
            "entities": [],
            "twelvelabs_people": None,
            "twelvelabs_objects": None,
            "twelvelabs_scene_summary": None,
            "video_metadata": None,
            "total_face_detections": len(disk["unique_faces"]),
            "total_object_detections": len(disk["unique_objects"]),
        }
    return None


def run_redaction(
    job_id,
    face_encodings=None,
    object_classes=None,
    entity_ids=None,
    custom_regions=None,
    blur_strength=DEFAULT_BLUR_STRENGTH,
    detect_every_n=DEFAULT_DETECT_EVERY_N,
    detect_every_seconds=None,
    use_temporal_optimization=True,
):
    job = get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    if job["status"] not in ("ready",):
        raise ValueError(f"Job {job_id} is not ready (status={job['status']})")

    video_path = job["video_path"]
    video_id = job.get("twelvelabs_video_id")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"redacted_{job_id}_{run_id}.mp4")

    temporal_ranges = []

    if use_temporal_optimization and entity_ids and video_id:
        for eid in entity_ids:
            try:
                ranges = twelvelabs_service.entity_search_time_ranges(
                    entity_id=eid, video_id=video_id,
                )
                temporal_ranges.extend(ranges)
                logger.info("Entity %s: found %d temporal ranges", eid, len(ranges))
            except Exception as e:
                logger.warning("Entity search failed for %s: %s", eid, str(e))

    if not temporal_ranges and use_temporal_optimization and face_encodings:
        if video_id:
            try:
                people = job.get("twelvelabs_people", [])
                if isinstance(people, list):
                    for person in people:
                        if isinstance(person, dict) and person.get("time_ranges"):
                            for tr in person["time_ranges"]:
                                temporal_ranges.append({
                                    "start": tr.get("start_sec", 0),
                                    "end": tr.get("end_sec", 9999),
                                })
            except Exception:
                pass

    if not face_encodings and entity_ids:
        unique_faces = job.get("unique_faces", [])
        for face in unique_faces:
            face_entity_id = face.get("entity_id")
            if face_entity_id and face_entity_id in entity_ids:
                enc = face.get("encoding")
                if enc:
                    face_encodings = face_encodings or []
                    face_encodings.append(enc)

    enc_arrays = [np.array(e) for e in (face_encodings or [])]
    obj_set = set(object_classes or [])

    result = redact_video(
        input_path=video_path,
        output_path=output_path,
        face_encodings=enc_arrays,
        object_classes=obj_set,
        blur_strength=blur_strength,
        detect_every_n=detect_every_n,
        detect_every_seconds=detect_every_seconds,
        temporal_ranges=temporal_ranges if temporal_ranges else None,
        custom_regions=custom_regions or [],
    )

    result["download_url"] = f"/api/download/{os.path.basename(output_path)}"
    result["entity_ids_used"] = entity_ids or []
    result["temporal_ranges_from_entity_search"] = len(temporal_ranges)
    return result
