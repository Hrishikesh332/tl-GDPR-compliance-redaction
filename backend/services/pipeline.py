import logging
import os
import subprocess
import threading
import uuid
from datetime import datetime, timezone

import numpy as np

from config import (
    OUTPUT_DIR, KEYFRAME_INTERVAL_SEC,
    DEFAULT_BLUR_STRENGTH, DEFAULT_DETECT_EVERY_N,
)
from services import twelvelabs_service
from services.detection import detect_faces
from services.clustering import cluster_faces
from services.face_identity import ensure_face_identity, get_face_identity
from services.redactor import redact_video
from utils.video import (
    extract_keyframes, extract_frames_at_timestamps,
    get_video_metadata, merge_overlapping_ranges, timestamps_from_time_ranges,
    normalize_export_height,
)
from utils.downloads import safe_redacted_mp4_filename
from utils.storage import (
    get_run_dir,
    save_unique_face_snaps,
    save_detection_metadata,
    load_faces_objects_from_disk,
    save_job_manifest,
    load_job_manifest,
    list_run_ids,
)
logger = logging.getLogger("video_redaction.pipeline")

jobs = {}
jobs_lock = threading.Lock()
manifests_cleaned = False
FACE_DESCRIPTION_MIN_MATCH_SCORE = 4.0
FACE_PRIVACY_MIN_CONFIDENCE = 0.68

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


def cleanup_orphan_temp_files():
    """Remove incomplete temp files left behind by interrupted pipelines."""
    if not os.path.isdir(OUTPUT_DIR):
        return
    removed = 0
    for fn in os.listdir(OUTPUT_DIR):
        lower = fn.lower()
        if not lower.startswith("tmp"):
            continue
        if not lower.endswith(VIDEO_EXTENSIONS):
            continue
        path = os.path.join(OUTPUT_DIR, fn)
        if not os.path.isfile(path):
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    if removed:
        logger.info("Cleaned up %d orphan temp files from output directory", removed)


def cleanup_duplicate_video_id_mappings():
    """Ensure each ``twelvelabs_video_id`` is claimed by at most one job.

    When multiple jobs share the same video_id (usually caused by the old
    heuristic recovery), keep only the newest job's mapping and clear the
    rest.  This runs once on first lookup.
    """
    global manifests_cleaned
    if manifests_cleaned:
        return
    manifests_cleaned = True
    cleanup_orphan_temp_files()

    video_to_jobs = {}
    for jid in list_run_ids():
        manifest = load_job_manifest(jid)
        if not manifest:
            continue
        vid = str(manifest.get("twelvelabs_video_id") or "").strip()
        if not vid:
            continue
        mtime = run_dir_mtime(jid) or 0.0
        video_to_jobs.setdefault(vid, []).append((jid, mtime, manifest))

    for vid, entries in video_to_jobs.items():
        if len(entries) <= 1:
            continue
        entries.sort(key=lambda e: e[1], reverse=True)
        winner_jid = entries[0][0]
        for jid, entry_mtime, manifest in entries[1:]:
            logger.info(
                "Clearing stale video_id mapping: job %s had video_id %s (belongs to job %s)",
                jid, vid, winner_jid,
            )
            manifest["twelvelabs_video_id"] = ""
            run_dir = get_run_dir(jid)
            save_job_manifest(run_dir, manifest)


def download_video_from_hls(hls_url, video_id, filename=None):

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = filename or f"input_{video_id}.mp4"
    dest = os.path.join(OUTPUT_DIR, safe_name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        logger.info("Video already downloaded: %s", dest)
        return dest
    logger.info("Downloading video %s from HLS to %s", video_id, dest)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", hls_url, "-c", "copy", "-movflags", "+faststart", dest],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg HLS download failed (rc=%d): %s", result.returncode, (result.stderr or "")[-500:])
            if os.path.isfile(dest):
                os.remove(dest)
            return None
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            logger.info("Downloaded video to %s (%.1f MB)", dest, os.path.getsize(dest) / 1e6)
            return dest
        return None
    except FileNotFoundError:
        logger.error("ffmpeg not found — cannot download HLS stream")
        return None
    except Exception as e:
        logger.error("HLS download failed for %s: %s", video_id, e)
        if os.path.isfile(dest):
            os.remove(dest)
        return None


def parse_iso_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def run_dir_mtime(job_id):
    run_dir = get_run_dir(job_id)
    try:
        return os.path.getmtime(run_dir)
    except OSError:
        return None


def candidate_source_videos():
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
        if lower.startswith("redacted_") or lower.startswith("person_") or lower.startswith("tmp"):
            continue
        candidates.append(path)
    return sorted(candidates)


def infer_video_path_for_job(job_id, target_meta=None):
    run_mtime = run_dir_mtime(job_id)
    candidates = candidate_source_videos()
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


def infer_video_path_for_video(video_id, info=None):
    if not video_id:
        return None

    if info is None:
        try:
            info = twelvelabs_service.get_video_info(video_id)
        except Exception as e:
            logger.warning("Could not retrieve video info for %s while inferring local path: %s", video_id, e)
            info = {}

    target_meta = info.get("system_metadata") or {}
    target_filename = str(target_meta.get("filename") or "").strip().lower()
    target_time = parse_iso_timestamp(info.get("indexed_at")) or parse_iso_timestamp(info.get("created_at"))

    candidates = candidate_source_videos()
    if not candidates:
        return None

    target_width = target_meta.get("width")
    target_height = target_meta.get("height")
    target_duration = target_meta.get("duration") or target_meta.get("duration_sec")

    best_path = None
    best_score = float("inf")

    for path in candidates:
        try:
            stat = os.stat(path)
        except OSError:
            continue

        basename = os.path.basename(path).lower()
        score = abs(stat.st_mtime - target_time) if target_time is not None else 0.0

        if basename.startswith("index_"):
            score -= 60.0
        elif basename.startswith("upload_"):
            score -= 45.0
        elif basename.startswith("input_"):
            score -= 30.0

        if target_filename and basename == target_filename:
            score -= 240.0

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


def build_manifest(job_id, job=None, overrides=None):
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
        "twelvelabs_people": source.get("twelvelabs_people", manifest.get("twelvelabs_people")),
        "twelvelabs_objects": source.get("twelvelabs_objects", manifest.get("twelvelabs_objects")),
        "twelvelabs_scene_summary": source.get("twelvelabs_scene_summary", manifest.get("twelvelabs_scene_summary")),
    })
    if overrides:
        manifest.update(overrides)
    return manifest


def persist_job_manifest(job_id, job=None, overrides=None):
    manifest = build_manifest(job_id, job=job, overrides=overrides)
    run_dir = get_run_dir(job_id)
    save_job_manifest(run_dir, manifest)
    return manifest


def load_job_from_disk(job_id):
    manifest = load_job_manifest(job_id)
    disk = load_faces_objects_from_disk(job_id)
    run_mtime = run_dir_mtime(job_id)
    manifest_changed = False

    if manifest is None and disk is None:
        return None

    target_meta = (manifest or {}).get("video_metadata") or {}
    video_path = (manifest or {}).get("video_path")
    if video_path and not os.path.isfile(video_path):
        video_path = None
    if not video_path:
        video_path = infer_video_path_for_job(job_id, target_meta=target_meta)
        if video_path:
            manifest = manifest or {"job_id": job_id}
            manifest["video_path"] = video_path
            manifest_changed = True

    created_at = (manifest or {}).get("created_at")
    if not created_at and run_mtime is not None:
        created_at = datetime.fromtimestamp(run_mtime, tz=timezone.utc).isoformat()

    stored_status = (manifest or {}).get("status", "ready")
    if stored_status == "processing":
        if disk and disk.get("unique_faces"):
            stored_status = "ready"
        else:
            stored_status = "failed"
        logger.info(
            "Job %s was stuck in 'processing' on disk (server restart); "
            "recovering as '%s'", job_id, stored_status,
        )
        if manifest:
            manifest["status"] = stored_status
            manifest_changed = True

    job = {
        "status": stored_status,
        "video_path": video_path,
        "video_filename": (manifest or {}).get("video_filename"),
        "created_at": created_at,
        "twelvelabs_video_id": (manifest or {}).get("twelvelabs_video_id"),
        "twelvelabs_status": (manifest or {}).get("twelvelabs_status", "done"),
        "local_status": (manifest or {}).get("local_status", "done"),
        "video_metadata": (manifest or {}).get("video_metadata"),
        "twelvelabs_people": (manifest or {}).get("twelvelabs_people"),
        "twelvelabs_objects": (manifest or {}).get("twelvelabs_objects"),
        "twelvelabs_scene_summary": (manifest or {}).get("twelvelabs_scene_summary"),
        "unique_faces": disk["unique_faces"] if disk else [],
        "unique_objects": disk["unique_objects"] if disk else [],
        "total_face_detections": len(disk["unique_faces"]) if disk else 0,
        "total_object_detections": len(disk["unique_objects"]) if disk else 0,
    }

    if manifest and manifest_changed:
        persist_job_manifest(job_id, overrides=manifest)
    return job


def recover_job_id_for_video(video_id):
    """Try to find an existing job that belongs to this video.

    Only returns a job when there is strong evidence of a match:
      - The manifest already has ``twelvelabs_video_id`` set to *this* video, OR
      - The manifest has NO ``twelvelabs_video_id`` yet AND the filename matches exactly.

    Never steals a job that is already bound to a *different* video.
    """
    if not video_id:
        return None

    try:
        info = twelvelabs_service.get_video_info(video_id)
    except Exception as e:
        logger.warning("Could not retrieve video info for %s during recovery: %s", video_id, e)
        info = {}

    target_meta = info.get("system_metadata") or {}
    target_filename = str(target_meta.get("filename") or "").strip().lower()
    if not target_filename and not target_meta:
        return None

    for job_id in list_run_ids():
        manifest = load_job_manifest(job_id) or {}
        existing_vid = str(manifest.get("twelvelabs_video_id") or "").strip()

        if existing_vid == video_id:
            return job_id

        if existing_vid:
            continue

        manifest_filename = str(manifest.get("video_filename") or "").strip().lower()
        if not target_filename or not manifest_filename:
            continue
        if manifest_filename != target_filename:
            continue

        recovered_manifest = {
            "job_id": job_id,
            "twelvelabs_video_id": video_id,
        }
        persist_job_manifest(job_id, overrides=recovered_manifest)
        logger.info(
            "Recovered unbound job %s for video %s (filename match: %s)",
            job_id, video_id, target_filename,
        )
        return job_id

    return None


def new_job_id():
    return str(uuid.uuid4())[:12]


def get_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if job:
        return job
    disk_job = load_job_from_disk(job_id)
    if disk_job:
        with jobs_lock:
            jobs[job_id] = disk_job
        return disk_job
    return None


def get_job_id_by_video_id(video_id):
    """Return the best explicitly-mapped job_id for a video, preferring the most recent run."""
    if not video_id:
        return None
    cleanup_duplicate_video_id_mappings()

    def candidate_key(job_id, source):
        created_at = parse_iso_timestamp((source or {}).get("created_at"))
        run_time = run_dir_mtime(job_id)
        effective_time = created_at if created_at is not None else (run_time if run_time is not None else 0.0)
        status = str((source or {}).get("status") or "")
        status_rank = {
            "ready": 3,
            "processing": 2,
            "failed": 1,
        }.get(status, 0)
        return (effective_time, status_rank, job_id)

    candidates = {}
    with jobs_lock:
        for jid, j in jobs.items():
            if j.get("twelvelabs_video_id") == video_id:
                candidates[jid] = dict(j)
    for jid in list_run_ids():
        manifest = load_job_manifest(jid)
        if manifest and manifest.get("twelvelabs_video_id") == video_id:
            if jid not in candidates:
                candidates[jid] = manifest
    if candidates:
        return max(candidates.items(), key=lambda item: candidate_key(item[0], item[1]))[0]
    return None


def get_exact_job_id_by_video_id(video_id):
    """Return a job_id only when the mapping to this video_id is explicit."""
    if not video_id:
        return None

    def candidate_key(job_id, source):
        created_at = parse_iso_timestamp((source or {}).get("created_at"))
        run_time = run_dir_mtime(job_id)
        effective_time = created_at if created_at is not None else (run_time if run_time is not None else 0.0)
        status = str((source or {}).get("status") or "")
        status_rank = {
            "ready": 3,
            "processing": 2,
            "failed": 1,
        }.get(status, 0)
        return (effective_time, status_rank, job_id)

    candidates = {}
    with jobs_lock:
        for jid, j in jobs.items():
            if j.get("twelvelabs_video_id") == video_id:
                candidates[jid] = dict(j)
    for jid in list_run_ids():
        manifest = load_job_manifest(jid)
        if manifest and manifest.get("twelvelabs_video_id") == video_id and jid not in candidates:
            candidates[jid] = manifest
    if candidates:
        return max(candidates.items(), key=lambda item: candidate_key(item[0], item[1]))[0]
    return None


def ensure_job_for_video(video_id, interval_sec=None, force=False):
    if not force:
        job_id = get_exact_job_id_by_video_id(video_id)
        if job_id:
            existing = get_job(job_id)
            if existing and existing.get("status") == "failed":
                logger.info(
                    "Existing job %s for video %s is failed; will create a fresh job",
                    job_id, video_id,
                )
                force = True
            else:
                return job_id

    try:
        info = twelvelabs_service.get_video_info(video_id)
    except Exception as e:
        logger.warning("Could not retrieve video info for %s while ensuring job: %s", video_id, e)
        info = {}

    video_path = infer_video_path_for_video(video_id, info=info)
    if not video_path or not os.path.isfile(video_path):
        hls_info = info.get("hls") or {}
        hls_url = hls_info.get("video_url")
        if hls_url:
            target_meta = info.get("system_metadata") or {}
            filename_hint = target_meta.get("filename")
            video_path = download_video_from_hls(hls_url, video_id, filename=filename_hint)
        if not video_path or not os.path.isfile(video_path):
            return None

    target_meta = info.get("system_metadata") or {}
    video_filename = target_meta.get("filename") or os.path.basename(video_path)

    logger.info(
        "Creating %slocal processing job for video %s using source %s",
        "fresh " if force else "",
        video_id,
        video_path,
    )
    return start_ingestion(
        video_path=video_path,
        video_filename=video_filename,
        interval_sec=interval_sec,
        skip_indexing=True,
        existing_video_id=video_id,
    )


def list_jobs():
    with jobs_lock:
        return [
            {
                "job_id": jid,
                "status": j["status"],
                "created_at": j.get("created_at"),
                "video_filename": j.get("video_filename"),
            }
            for jid, j in jobs.items()
        ]


def start_ingestion(video_path, video_filename=None, interval_sec=None, skip_indexing=False, existing_video_id=None):
    job_id = new_job_id()
    interval = interval_sec or KEYFRAME_INTERVAL_SEC

    with jobs_lock:
        jobs[job_id] = {
            "status": "processing",
            "video_path": video_path,
            "video_filename": video_filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "twelvelabs_video_id": existing_video_id,
            "twelvelabs_status": "pending",
            "local_status": "pending",
            "error": None,
        }
    persist_job_manifest(job_id, job=jobs[job_id])

    thread = threading.Thread(
        target=run_ingestion,
        args=(job_id, video_path, interval, skip_indexing, existing_video_id),
        daemon=True,
    )
    thread.start()
    return job_id


def run_ingestion(job_id, video_path, interval_sec, skip_indexing=False, existing_video_id=None):
    try:
        logger.info("[Job %s] Pipeline started (interval=%.1fs, skip_indexing=%s)", job_id, interval_sec, skip_indexing)

        logger.info("[Job %s] STEP 1/7: Getting video metadata...", job_id)
        metadata = get_video_metadata(video_path)
        with jobs_lock:
            jobs[job_id]["video_metadata"] = metadata
            snapshot = dict(jobs[job_id])
        persist_job_manifest(job_id, job=snapshot)
        logger.info("[Job %s] STEP 1/7: Done — %sx%s, %.1fs, %.0f fps", job_id,
                    metadata.get("width"), metadata.get("height"),
                    metadata.get("duration_sec"), metadata.get("fps"))

        video_id = (existing_video_id or "").strip() or None
        people_desc = []
        objects_desc = []
        scene_summary = {}

        if skip_indexing and video_id:
            logger.info("[Job %s] STEP 2/7: Skipping indexing — using existing video_id=%s", job_id, video_id)
            with jobs_lock:
                jobs[job_id]["twelvelabs_video_id"] = video_id
                jobs[job_id]["twelvelabs_status"] = "indexed"
                snapshot = dict(jobs[job_id])
            persist_job_manifest(job_id, job=snapshot)
        elif skip_indexing:
            logger.info("[Job %s] STEP 2/7: Skipping indexing (no video_id provided)", job_id)
            with jobs_lock:
                jobs[job_id]["twelvelabs_status"] = "skipped"
                snapshot = dict(jobs[job_id])
            persist_job_manifest(job_id, job=snapshot)
        else:
            logger.info("[Job %s] STEP 2/7: TwelveLabs — uploading and indexing video...", job_id)
            with jobs_lock:
                jobs[job_id]["twelvelabs_status"] = "uploading"

            try:
                tl_result = twelvelabs_service.ingest_video(video_path)
                video_id = tl_result["video_id"]
                task_id = tl_result["task_id"]

                with jobs_lock:
                    jobs[job_id]["twelvelabs_video_id"] = video_id
                    jobs[job_id]["twelvelabs_task_id"] = task_id
                    jobs[job_id]["twelvelabs_status"] = "indexed"
                    snapshot = dict(jobs[job_id])
                persist_job_manifest(job_id, job=snapshot)

                logger.info("[Job %s] STEP 2/7: Done — indexed, video_id=%s", job_id, video_id)

            except Exception as e:
                logger.warning("[Job %s] STEP 2/7: TwelveLabs indexing failed: %s (will use interval keyframes)", job_id, str(e))
                with jobs_lock:
                    jobs[job_id]["twelvelabs_status"] = f"index_failed: {str(e)}"
                    snapshot = dict(jobs[job_id])
                persist_job_manifest(job_id, job=snapshot)

        if video_id:
            logger.info("[Job %s] STEP 3/7: TwelveLabs — Pegasus 1.5 metadata (people, scenes)...", job_id)
            with jobs_lock:
                jobs[job_id]["twelvelabs_status"] = "pegasus_metadata"

            try:
                pegasus_metadata = twelvelabs_service.describe_video_with_pegasus(video_id)
                people_desc = pegasus_metadata.get("people", [])
                objects_desc = []
                scene_summary = pegasus_metadata.get("scene_summary", {})

                with jobs_lock:
                    jobs[job_id]["twelvelabs_people"] = people_desc
                    jobs[job_id]["twelvelabs_objects"] = objects_desc
                    jobs[job_id]["twelvelabs_scene_summary"] = scene_summary
                    jobs[job_id]["twelvelabs_pegasus_metadata_task_id"] = pegasus_metadata.get("task_id")
                    jobs[job_id]["twelvelabs_status"] = "pegasus_metadata_ready"
                    snapshot = dict(jobs[job_id])
                persist_job_manifest(job_id, job=snapshot)

                logger.info("[Job %s] STEP 3/7: Done — Pegasus 1.5 metadata complete", job_id)

            except Exception as e:
                logger.warning("[Job %s] STEP 3/7: Pegasus 1.5 metadata failed: %s", job_id, str(e))
                with jobs_lock:
                    jobs[job_id]["twelvelabs_status"] = f"pegasus_metadata_failed: {str(e)}"
                    snapshot = dict(jobs[job_id])
                persist_job_manifest(job_id, job=snapshot)
        else:
            logger.info("[Job %s] STEP 3/7: Skipped (no video_id)", job_id)

        logger.info("[Job %s] STEP 4/7: Extracting keyframes...", job_id)
        with jobs_lock:
            jobs[job_id]["local_status"] = "extracting_keyframes"

        analyzed_timestamps = collect_timestamps_from_analysis(
            people_desc, objects_desc
        )

        if analyzed_timestamps:
            keyframes = extract_frames_at_timestamps(video_path, analyzed_timestamps)
            logger.info("[Job %s] STEP 4/7: Done — %d keyframes from analysis timestamps", job_id, len(keyframes))
        else:
            keyframes = extract_keyframes(video_path, interval_sec=interval_sec)
            logger.info("[Job %s] STEP 4/7: Done — %d keyframes at %.1fs interval (no analysis timestamps)", job_id, len(keyframes), interval_sec)

        logger.info("[Job %s] STEP 5/7: Face detection on %d keyframes...", job_id, len(keyframes))
        with jobs_lock:
            jobs[job_id]["local_status"] = "detecting"

        all_faces = []
        all_objects = []

        for kf in keyframes:
            faces = detect_faces(kf["frame"], with_encodings=True)
            for f in faces:
                f["frame_idx"] = kf["frame_idx"]
                f["timestamp"] = kf["timestamp"]
                all_faces.append(f)

        logger.info("[Job %s] STEP 5/7: Done — %d face detections, object detection disabled", job_id, len(all_faces))

        logger.info("[Job %s] STEP 6/7: Clustering into unique faces...", job_id)
        with jobs_lock:
            jobs[job_id]["local_status"] = "clustering"

        unique_faces = cluster_faces(all_faces)
        unique_objects = []
        logger.info("[Job %s] STEP 6/7: Done — %d unique faces, %d unique objects", job_id, len(unique_faces), len(unique_objects))

        logger.info("[Job %s] STEP 7/7: Saving snapshots to disk...", job_id)
        run_dir = get_run_dir(job_id)
        save_unique_face_snaps(run_dir, unique_faces)

        enrich_faces_with_descriptions(job_id, people_desc, unique_faces)
        save_detection_metadata(run_dir, unique_faces, unique_objects)

        with jobs_lock:
            jobs[job_id]["local_status"] = "done"
            jobs[job_id]["unique_faces"] = unique_faces
            jobs[job_id]["unique_objects"] = unique_objects
            jobs[job_id]["total_face_detections"] = len(all_faces)
            jobs[job_id]["total_object_detections"] = len(all_objects)
            jobs[job_id]["status"] = "ready"
            if video_id and "failed" not in jobs[job_id].get("twelvelabs_status", ""):
                jobs[job_id]["twelvelabs_status"] = "done"
            snapshot = dict(jobs[job_id])
        persist_job_manifest(job_id, job=snapshot)
        logger.info("[Job %s] STEP 7/7: Done — snapshots saved", job_id)

        logger.info("[Job %s] Pipeline complete — status=ready", job_id)

    except Exception as e:
        logger.error("[Job %s] Pipeline failed: %s", job_id, str(e), exc_info=True)
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            snapshot = dict(jobs[job_id])
        persist_job_manifest(job_id, job=snapshot)


def collect_timestamps_from_analysis(people_desc, objects_desc):
    """Extract smart keyframe timestamps from Pegasus/person time ranges.

    Object detection is intentionally disabled in the ingestion pipeline, but
    the optional ``objects_desc`` argument is kept for older persisted metadata.
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


def normalize_analysis_time_ranges(time_ranges):
    normalized = []
    for time_range in time_ranges or []:
        if not isinstance(time_range, dict):
            continue
        try:
            start = float(time_range.get("start_sec", time_range.get("start")))
            end = float(time_range.get("end_sec", time_range.get("end")))
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        normalized.append((start, end))
    return normalized


def collect_temporal_ranges_from_face_targets(face_targets):
    ranges = []
    for face in face_targets or []:
        if not isinstance(face, dict):
            continue

        for appearance in face.get("appearances", []) or []:
            if not isinstance(appearance, dict):
                continue
            try:
                timestamp = float(appearance.get("timestamp"))
            except (TypeError, ValueError):
                continue
            ranges.append((max(0.0, timestamp - 1.0), max(0.0, timestamp + 1.35)))

    return [
        {"start": start, "end": end}
        for start, end in merge_overlapping_ranges(ranges)
    ]


def extract_face_appearance_timestamps(face):
    timestamps = []
    for appearance in face.get("appearances", []) or []:
        if not isinstance(appearance, dict):
            continue
        try:
            timestamp = float(appearance.get("timestamp"))
        except (TypeError, ValueError):
            continue
        timestamps.append(timestamp)
    if timestamps:
        return timestamps

    try:
        timestamp = float(face.get("timestamp"))
    except (TypeError, ValueError):
        return []
    return [timestamp]


def face_description_overlap_score(face, desc_entry):
    timestamps = extract_face_appearance_timestamps(face)
    time_ranges = normalize_analysis_time_ranges(desc_entry.get("time_ranges", []))
    if not timestamps or not time_ranges:
        return 0.0

    padded_ranges = [(start - 0.45, end + 0.45) for start, end in time_ranges]
    inside_count = 0
    hit_ranges = 0
    for start, end in padded_ranges:
        range_hit = False
        for timestamp in timestamps:
            if start <= timestamp <= end:
                inside_count += 1
                range_hit = True
        if range_hit:
            hit_ranges += 1

    if inside_count <= 0:
        return 0.0

    coverage = inside_count / max(len(timestamps), 1)
    range_coverage = hit_ranges / max(len(time_ranges), 1)
    return inside_count * 4.0 + coverage * 2.5 + range_coverage


def safe_float(value, default=None):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value == value else default
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def match_people_descriptions_to_faces(faces, people_desc):
    if not isinstance(people_desc, list) or not faces:
        return {}

    scored_pairs = []
    for face_index, face in enumerate(faces):
        for desc_index, desc_entry in enumerate(people_desc):
            if not isinstance(desc_entry, dict):
                continue
            score = face_description_overlap_score(face, desc_entry)
            if score >= FACE_DESCRIPTION_MIN_MATCH_SCORE:
                scored_pairs.append((score, face_index, desc_index))

    scored_pairs.sort(key=lambda item: (-item[0], item[2], item[1]))

    assignments = {}
    used_faces = set()
    used_desc = set()
    for score, face_index, desc_index in scored_pairs:
        if face_index in used_faces or desc_index in used_desc:
            continue
        assignments[face_index] = {"desc_index": desc_index, "score": score}
        used_faces.add(face_index)
        used_desc.add(desc_index)

    return assignments


def normalize_face_tags(desc_entry):
    should_anonymize = bool(desc_entry.get("should_anonymize"))
    is_official = bool(desc_entry.get("is_official"))
    confidence = safe_float(desc_entry.get("confidence"), None)
    review_required = bool(desc_entry.get("review_required"))
    if confidence is not None and confidence < FACE_PRIVACY_MIN_CONFIDENCE:
        review_required = True
    if review_required:
        should_anonymize = False
    tags = []
    seen = set()

    for raw_tag in desc_entry.get("tags", []) or []:
        tag = str(raw_tag or "").strip()
        if not tag:
            continue
        normalized_key = tag.lower()
        normalized_tag = tag
        if normalized_key == "anonymized":
            if not should_anonymize or is_official:
                continue
            normalized_tag = "Anonymized"
        elif normalized_key == "official":
            if not is_official:
                continue
            normalized_tag = "Official"
        elif normalized_key == "review":
            if not review_required:
                continue
            normalized_tag = "Review"
        if normalized_key in seen:
            continue
        tags.append(normalized_tag)
        seen.add(normalized_key)

    if should_anonymize and not is_official and "anonymized" not in seen:
        tags.insert(0, "Anonymized")
        seen.add("anonymized")
    if is_official and "official" not in seen:
        tags.append("Official")
    if review_required and "review" not in seen:
        tags.append("Review")

    return tags, should_anonymize and not is_official, is_official, review_required


def faces_need_description_refresh(faces, people_desc):
    if not isinstance(people_desc, list) or not faces:
        return False

    missing_fields = any(
        "priority_rank" not in face or
        "should_anonymize" not in face or
        "is_official" not in face or
        "review_required" not in face or
        "description_match_score" not in face or
        "tags" not in face
        for face in faces
        if isinstance(face, dict)
    )
    if missing_fields:
        return True

    has_people_privacy_annotations = any(
        isinstance(desc_entry, dict) and (
            bool(desc_entry.get("should_anonymize")) or
            bool(desc_entry.get("is_official")) or
            bool(desc_entry.get("tags"))
        )
        for desc_entry in people_desc
    )
    if not has_people_privacy_annotations:
        return False

    has_face_privacy_annotations = any(
        isinstance(face, dict) and (
            bool(face.get("should_anonymize")) or
            bool(face.get("is_official")) or
            any(str(tag or "").strip().lower() in ("anonymized", "official") for tag in face.get("tags", []) or [])
        )
        for face in faces
    )
    return not has_face_privacy_annotations


def enrich_faces_with_descriptions(job_id, people_desc, unique_faces=None, preserve_redaction_decisions=False):
    faces = unique_faces
    if faces is None:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job or not job.get("unique_faces"):
                return
            faces = job["unique_faces"]

    if not faces:
        return

    if isinstance(people_desc, list):
        for i, face in enumerate(faces):
            ensure_face_identity(face, fallback_index=i)
            if preserve_redaction_decisions:
                face.setdefault("tags", [])
                face.setdefault("should_anonymize", False)
                face.setdefault("is_official", False)
                face.setdefault("review_required", False)
            else:
                face["tags"] = []
                face["should_anonymize"] = False
                face["is_official"] = False
                face["review_required"] = False
            face["description_match_score"] = 0.0

    if isinstance(people_desc, list):
        assignments = match_people_descriptions_to_faces(faces, people_desc)
        for i, face in enumerate(faces):
            ensure_face_identity(face, fallback_index=i)
            assignment = assignments.get(i)
            if isinstance(assignment, dict):
                desc_index = assignment.get("desc_index")
                face["description_match_score"] = round(float(assignment.get("score") or 0.0), 3)
            else:
                desc_index = assignment
            if desc_index is None or desc_index >= len(people_desc):
                if not preserve_redaction_decisions:
                    face["review_required"] = True
                    face["tags"] = ["Review"]
                continue
            desc_entry = people_desc[desc_index]
            if not isinstance(desc_entry, dict):
                continue
            face["description"] = desc_entry.get("description", "")
            face["time_ranges"] = desc_entry.get("time_ranges", [])
            face["priority_rank"] = desc_index
            confidence = safe_float(desc_entry.get("confidence"), None)
            if confidence is not None:
                face["description_confidence"] = round(confidence, 3)
            if desc_entry.get("redaction_reason"):
                face["redaction_reason"] = desc_entry.get("redaction_reason")
            tags, should_anonymize, is_official, review_required = normalize_face_tags(desc_entry)
            if not preserve_redaction_decisions:
                face["tags"] = tags
                face["should_anonymize"] = should_anonymize
                face["is_official"] = is_official
                face["review_required"] = review_required
            raw_name = desc_entry.get("name") or desc_entry.get("person_name") or ""
            if raw_name:
                face["name"] = raw_name

    for i, face in enumerate(faces):
        stable_person_id = ensure_face_identity(face, fallback_index=i)
        if not face.get("name"):
            face["name"] = stable_person_id or f"person_{i}"
        if "priority_rank" not in face:
            face["priority_rank"] = len(people_desc) + i if isinstance(people_desc, list) else i
        if "tags" not in face:
            face["tags"] = []
        if "should_anonymize" not in face:
            face["should_anonymize"] = False
        if "is_official" not in face:
            face["is_official"] = False
        if "review_required" not in face:
            face["review_required"] = False


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
    with jobs_lock:
        jobs[job_id]["entities"] = entities
    save_detection_metadata(run_dir, unique_faces, job.get("unique_objects", []))
    logger.info("Job %s: pushed %d entities to TwelveLabs (user-requested)", job_id, len(entities))
    return entities


def get_enriched_faces(job_id):
    job = get_job(job_id)
    if job:
        people_desc = job.get("twelvelabs_people")
        faces = job.get("unique_faces", [])
        for index, face in enumerate(faces):
            ensure_face_identity(face, fallback_index=index)
        video_id = str(job.get("twelvelabs_video_id") or "").strip()
        if faces and not isinstance(people_desc, list) and video_id:
            try:
                pegasus_metadata = twelvelabs_service.describe_video_with_pegasus(video_id)
                people_desc = pegasus_metadata.get("people", [])
            except Exception as e:
                logger.warning("Could not refresh Pegasus 1.5 people metadata for job %s (%s): %s", job_id, video_id, e)
            else:
                job["twelvelabs_people"] = people_desc
                job["twelvelabs_objects"] = []
                job["twelvelabs_scene_summary"] = pegasus_metadata.get("scene_summary", {})
                job["twelvelabs_pegasus_metadata_task_id"] = pegasus_metadata.get("task_id")
        if faces and isinstance(people_desc, list):
            if faces_need_description_refresh(faces, people_desc):
                enrich_faces_with_descriptions(
                    job_id,
                    people_desc,
                    faces,
                    preserve_redaction_decisions=True,
                )
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
            "error": job.get("error"),
        }
    disk = load_faces_objects_from_disk(job_id)
    if disk:
        for index, face in enumerate(disk["unique_faces"]):
            ensure_face_identity(face, fallback_index=index)
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
            "error": None,
        }
    return None


def run_redaction(
    job_id,
    face_encodings=None,
    face_targets=None,
    focus_face_targets=None,
    object_classes=None,
    entity_ids=None,
    custom_regions=None,
    blur_strength=DEFAULT_BLUR_STRENGTH,
    redaction_style="blur",
    detect_every_n=DEFAULT_DETECT_EVERY_N,
    detect_every_seconds=None,
    use_temporal_optimization=True,
    output_height=720,
    reverse_face_redaction=False,
    progress_callback=None,
):
    job = get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    if job["status"] not in ("ready",):
        raise ValueError(f"Job {job_id} is not ready (status={job['status']})")

    video_path = job["video_path"]
    video_id = job.get("twelvelabs_video_id")
    normalized_output_height = normalize_export_height(output_height)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"redacted_{job_id}_{run_id}_{normalized_output_height}p.mp4")

    temporal_ranges = []

    if use_temporal_optimization and face_targets:
        temporal_ranges = collect_temporal_ranges_from_face_targets(face_targets)
        if temporal_ranges:
            logger.info(
                "Selected face targets limited redaction to %d temporal ranges",
                len(temporal_ranges),
            )

    if not temporal_ranges and use_temporal_optimization and entity_ids and video_id:
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

    focus_face_targets = focus_face_targets or []
    if reverse_face_redaction:
        face_encodings = ["__ALL__"]
        face_targets = []
        object_classes = []
        use_temporal_optimization = False
        detect_every_n = 1

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

    obj_set = set()

    face_lock_tracks = {}
    face_lock_failures = []
    if face_targets and not reverse_face_redaction:
        from services.face_lock_track import build_face_lock_lane

        for face in face_targets:
            person_id = get_face_identity(face)
            if not person_id:
                continue
            label = (
                face.get("name")
                or face.get("description")
                or person_id
            )
            if face.get("is_snapped_entity"):
                face_lock_failures.append({
                    "person_id": person_id,
                    "label": label,
                    "reason": "Snapped face has no anchor history; falling back to per-frame match",
                    "fallback": "per_frame",
                })
                continue
            try:
                lane_doc = build_face_lock_lane(job_id, person_id)
                if lane_doc:
                    face_lock_tracks[person_id] = lane_doc
                    logger.info(
                        "Face-lock lane ready for person %s: %d frames",
                        person_id, len(lane_doc.get("lane") or []),
                    )
                else:
                    face_lock_failures.append({
                        "person_id": person_id,
                        "label": label,
                        "reason": "Face-lock build returned empty lane",
                        "fallback": "per_frame",
                    })
            except Exception as exc:
                logger.warning(
                    "Could not build face-lock lane for %s; falling back to per-frame relock: %s",
                    person_id, exc,
                )
                face_lock_failures.append({
                    "person_id": person_id,
                    "label": label,
                    "reason": str(exc) or "Face-lock build failed",
                    "fallback": "per_frame",
                })

    preserve_face_lock_tracks = {}
    if reverse_face_redaction and focus_face_targets:
        from services.face_lock_track import build_face_lock_lane

        for face in focus_face_targets:
            if face.get("is_snapped_entity"):
                continue
            person_id = get_face_identity(face)
            if not person_id:
                continue
            try:
                lane_doc = build_face_lock_lane(job_id, person_id)
                if lane_doc:
                    preserve_face_lock_tracks[person_id] = lane_doc
                    logger.info(
                        "Reverse focus face-lock lane ready for person %s: %d frames",
                        person_id, len(lane_doc.get("lane") or []),
                    )
            except Exception as exc:
                logger.warning(
                    "Could not build reverse focus face-lock lane for %s; falling back to per-frame focus localization: %s",
                    person_id, exc,
                )

    result = redact_video(
        input_path=video_path,
        output_path=output_path,
        face_encodings=enc_arrays,
        face_targets=face_targets or [],
        object_classes=obj_set,
        blur_strength=blur_strength,
        redaction_style=redaction_style,
        detect_every_n=detect_every_n,
        detect_every_seconds=detect_every_seconds,
        temporal_ranges=temporal_ranges if temporal_ranges else None,
        custom_regions=custom_regions or [],
        output_height=normalized_output_height,
        progress_callback=progress_callback,
        face_lock_tracks=face_lock_tracks,
        reverse_face_redaction=reverse_face_redaction,
        preserve_face_targets=focus_face_targets,
        preserve_face_lock_tracks=preserve_face_lock_tracks,
    )

    download_filename = safe_redacted_mp4_filename(os.path.basename(output_path))
    result["download_url"] = f"/api/download/{download_filename}"
    result["download_filename"] = download_filename
    result["mime_type"] = "video/mp4"
    result["export_quality"] = f"{normalized_output_height}p"
    result["entity_ids_used"] = entity_ids or []
    result["temporal_ranges_from_entity_search"] = len(temporal_ranges)
    result["face_lock_failures"] = face_lock_failures
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
