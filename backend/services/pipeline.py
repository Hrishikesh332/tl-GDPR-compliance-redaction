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
from services.detection import detect_faces, detect_objects
from services.clustering import cluster_faces, cluster_objects
from services.redactor import redact_video
from utils.video import (
    extract_keyframes, extract_frames_at_timestamps,
    get_video_metadata, timestamps_from_time_ranges,
)
from utils.storage import (
    get_run_dir,
    save_unique_face_snaps,
    save_unique_object_snaps,
    load_faces_objects_from_disk,
    save_job_manifest,
    load_job_manifest,
    list_run_ids,
)
logger = logging.getLogger("video_redaction.pipeline")

_jobs = {}
_lock = threading.Lock()

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


def _parse_iso_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _run_dir_mtime(job_id):
    run_dir = get_run_dir(job_id)
    try:
        return os.path.getmtime(run_dir)
    except OSError:
        return None


def _candidate_source_videos():
    if not os.path.isdir(OUTPUT_DIR):
        return []
    candidates = []
    for fn in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, fn)
        if not os.path.isfile(path):
            continue
        lower = fn.lower()
        if not lower.endswith(VIDEO_EXTENSIONS):
            continue
        if lower.startswith("redacted_") or lower.startswith("person_"):
            continue
        candidates.append(path)
    return sorted(candidates)


def _infer_video_path_for_job(job_id, target_meta=None):
    run_mtime = _run_dir_mtime(job_id)
    candidates = _candidate_source_videos()
    if not candidates:
        return None

    target_width = (target_meta or {}).get("width")
    target_height = (target_meta or {}).get("height")
    target_duration = (target_meta or {}).get("duration") or (target_meta or {}).get("duration_sec")

    best_path = None
    best_score = float("inf")

    for path in candidates:
        try:
            stat = os.stat(path)
        except OSError:
            continue

        score = abs(stat.st_mtime - run_mtime) if run_mtime is not None else 0.0
        if os.path.basename(path).startswith("input_"):
            score -= 45.0

        if target_width or target_height or target_duration:
            try:
                meta = get_video_metadata(path)
                if target_width and meta.get("width"):
                    score += abs(meta["width"] - target_width) * 1.5
                if target_height and meta.get("height"):
                    score += abs(meta["height"] - target_height) * 1.5
                if target_duration and meta.get("duration_sec"):
                    score += abs(meta["duration_sec"] - target_duration) * 30.0
            except Exception:
                score += 1e6

        if score < best_score:
            best_score = score
            best_path = path

    return best_path


def _build_manifest(job_id, job=None, overrides=None):
    manifest = dict(load_job_manifest(job_id) or {})
    source = job or {}
    manifest.update({
        "job_id": job_id,
        "status": source.get("status", manifest.get("status", "ready")),
        "created_at": source.get("created_at", manifest.get("created_at")),
        "video_path": source.get("video_path", manifest.get("video_path")),
        "video_filename": source.get("video_filename", manifest.get("video_filename")),
        "twelvelabs_video_id": source.get("twelvelabs_video_id", manifest.get("twelvelabs_video_id")),
        "video_metadata": source.get("video_metadata", manifest.get("video_metadata")),
        "twelvelabs_status": source.get("twelvelabs_status", manifest.get("twelvelabs_status")),
        "local_status": source.get("local_status", manifest.get("local_status")),
    })
    if overrides:
        manifest.update(overrides)
    return manifest


def _persist_job_manifest(job_id, job=None, overrides=None):
    manifest = _build_manifest(job_id, job=job, overrides=overrides)
    run_dir = get_run_dir(job_id)
    save_job_manifest(run_dir, manifest)
    return manifest


def _load_job_from_disk(job_id):
    manifest = load_job_manifest(job_id)
    disk = load_faces_objects_from_disk(job_id)
    run_mtime = _run_dir_mtime(job_id)

    if manifest is None and disk is None:
        return None

    target_meta = (manifest or {}).get("video_metadata") or {}
    video_path = (manifest or {}).get("video_path")
    if video_path and not os.path.isfile(video_path):
        video_path = None
    if not video_path:
        video_path = _infer_video_path_for_job(job_id, target_meta=target_meta)
        if video_path:
            manifest = manifest or {"job_id": job_id}
            manifest["video_path"] = video_path

    created_at = (manifest or {}).get("created_at")
    if not created_at and run_mtime is not None:
        created_at = datetime.fromtimestamp(run_mtime, tz=timezone.utc).isoformat()

    job = {
        "status": (manifest or {}).get("status", "ready"),
        "video_path": video_path,
        "video_filename": (manifest or {}).get("video_filename"),
        "created_at": created_at,
        "twelvelabs_video_id": (manifest or {}).get("twelvelabs_video_id"),
        "twelvelabs_status": (manifest or {}).get("twelvelabs_status", "done"),
        "local_status": (manifest or {}).get("local_status", "done"),
        "video_metadata": (manifest or {}).get("video_metadata"),
        "unique_faces": disk["unique_faces"] if disk else [],
        "unique_objects": disk["unique_objects"] if disk else [],
        "total_face_detections": len(disk["unique_faces"]) if disk else 0,
        "total_object_detections": len(disk["unique_objects"]) if disk else 0,
    }

    if manifest:
        _persist_job_manifest(job_id, overrides=manifest)
    return job


def _recover_job_id_for_video(video_id):
    try:
        info = twelvelabs_service.get_video_info(video_id)
    except Exception as e:
        logger.warning("Could not retrieve video info for %s during recovery: %s", video_id, e)
        info = {}

    target_meta = info.get("system_metadata") or {}
    target_filename = target_meta.get("filename")
    target_time = _parse_iso_timestamp(info.get("indexed_at")) or _parse_iso_timestamp(info.get("created_at"))
    if target_time is None and not target_filename and not target_meta:
        return None

    best_job_id = None
    best_score = float("inf")

    for job_id in list_run_ids():
        manifest = load_job_manifest(job_id) or {}
        if manifest.get("twelvelabs_video_id") == video_id:
            return job_id

        score = 0.0
        run_time = _run_dir_mtime(job_id)
        if target_time is not None and run_time is not None:
            score += abs(run_time - target_time)
        if target_filename:
            filename = manifest.get("video_filename")
            if filename:
                score += 0 if filename == target_filename else 3600.0
            else:
                score += 900.0

        inferred_path = manifest.get("video_path")
        if not inferred_path or not os.path.isfile(inferred_path):
            inferred_path = _infer_video_path_for_job(job_id, target_meta=target_meta)
        if inferred_path:
            score -= 120.0
        else:
            score += 1e6

        if score < best_score:
            best_score = score
            best_job_id = job_id

    if not best_job_id:
        return None

    recovered_manifest = {
        "job_id": best_job_id,
        "status": "ready",
        "created_at": datetime.fromtimestamp(_run_dir_mtime(best_job_id), tz=timezone.utc).isoformat() if _run_dir_mtime(best_job_id) else None,
        "video_path": _infer_video_path_for_job(best_job_id, target_meta=target_meta),
        "video_filename": target_filename,
        "twelvelabs_video_id": video_id,
        "video_metadata": target_meta or None,
        "twelvelabs_status": "done",
        "local_status": "done",
    }
    _persist_job_manifest(best_job_id, overrides=recovered_manifest)
    logger.info("Recovered disk job %s for video %s using persisted snaps/output files", best_job_id, video_id)
    return best_job_id


def _new_job_id():
    return str(uuid.uuid4())[:12]


def get_job(job_id):
    with _lock:
        job = _jobs.get(job_id)
    if job:
        return job
    disk_job = _load_job_from_disk(job_id)
    if disk_job:
        with _lock:
            _jobs[job_id] = disk_job
        return disk_job
    return None


def get_job_id_by_video_id(video_id):
    """Return the first job_id whose twelvelabs_video_id matches, or None."""
    if not video_id:
        return None
    with _lock:
        for jid, j in _jobs.items():
            if j.get("twelvelabs_video_id") == video_id:
                return jid
    for jid in list_run_ids():
        manifest = load_job_manifest(jid)
        if manifest and manifest.get("twelvelabs_video_id") == video_id:
            return jid
    return _recover_job_id_for_video(video_id)


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
    _persist_job_manifest(job_id, job=_jobs[job_id])

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
            snapshot = dict(_jobs[job_id])
        _persist_job_manifest(job_id, job=snapshot)
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
                snapshot = dict(_jobs[job_id])
            _persist_job_manifest(job_id, job=snapshot)
        elif skip_indexing:
            logger.info("[Job %s] STEP 2/7: Skipping indexing (no video_id provided)", job_id)
            with _lock:
                _jobs[job_id]["twelvelabs_status"] = "skipped"
                snapshot = dict(_jobs[job_id])
            _persist_job_manifest(job_id, job=snapshot)
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
                    snapshot = dict(_jobs[job_id])
                _persist_job_manifest(job_id, job=snapshot)

                logger.info("[Job %s] STEP 2/7: Done — indexed, video_id=%s", job_id, video_id)

            except Exception as e:
                logger.warning("[Job %s] STEP 2/7: TwelveLabs indexing failed: %s (will use interval keyframes)", job_id, str(e))
                with _lock:
                    _jobs[job_id]["twelvelabs_status"] = f"index_failed: {str(e)}"
                    snapshot = dict(_jobs[job_id])
                _persist_job_manifest(job_id, job=snapshot)

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
                    snapshot = dict(_jobs[job_id])
                _persist_job_manifest(job_id, job=snapshot)

                logger.info("[Job %s] STEP 3/7: Done — analysis complete", job_id)

            except Exception as e:
                logger.warning("[Job %s] STEP 3/7: TwelveLabs analysis failed: %s", job_id, str(e))
                with _lock:
                    _jobs[job_id]["twelvelabs_status"] = f"analysis_failed: {str(e)}"
                    snapshot = dict(_jobs[job_id])
                _persist_job_manifest(job_id, job=snapshot)
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
            snapshot = dict(_jobs[job_id])
        _persist_job_manifest(job_id, job=snapshot)
        logger.info("[Job %s] STEP 7/7: Done — snapshots saved", job_id)

        logger.info("[Job %s] Pipeline complete — status=ready", job_id)

    except Exception as e:
        logger.error("[Job %s] Pipeline failed: %s", job_id, str(e), exc_info=True)
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            snapshot = dict(_jobs[job_id])
        _persist_job_manifest(job_id, job=snapshot)


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

    auto_face_mode = (
        isinstance(face_encodings, list)
        and len(face_encodings) == 1
        and isinstance(face_encodings[0], str)
        and face_encodings[0] == "__ALL__"
    )

    if auto_face_mode:
        enc_arrays = face_encodings
    else:
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


def preview_redaction_tracks(job_id, custom_regions=None, preview_fps=8):
    job = get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    if job["status"] not in ("ready",):
        raise ValueError(f"Job {job_id} is not ready (status={job['status']})")

    video_path = job["video_path"]
    result = redact_video(
        input_path=video_path,
        output_path=None,
        custom_regions=custom_regions or [],
        collect_custom_track_data=True,
        track_sample_fps=preview_fps,
        preview_only=True,
    )
    return {
        "custom_tracks": result.get("custom_tracks", []),
        "fps": result.get("fps"),
        "width": result.get("width"),
        "height": result.get("height"),
        "total_frames": result.get("total_frames"),
    }
