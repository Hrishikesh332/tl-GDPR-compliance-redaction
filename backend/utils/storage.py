import base64
import json
import math
import os
import re

from config import SNAPS_DIR, OUTPUT_DIR

JOB_MANIFEST_FILENAME = "job_manifest.json"
DETECTION_METADATA_FILENAME = "detection_metadata.json"


def ensure_dirs():
    os.makedirs(SNAPS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_filename(name, index=None):
    s = re.sub(r"[^\w\-.]", "_", name).strip("_") or "item"
    return f"{s}_{index}" if index is not None else s


def get_run_dir(run_id):
    run_dir = os.path.join(SNAPS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def list_run_ids():
    if not os.path.isdir(SNAPS_DIR):
        return []
    return [
        name for name in sorted(os.listdir(SNAPS_DIR))
        if os.path.isdir(os.path.join(SNAPS_DIR, name))
    ]


def save_job_manifest(run_dir, manifest):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, JOB_MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)
    return path


def load_job_manifest(job_id):
    path = os.path.join(SNAPS_DIR, job_id, JOB_MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_snap(directory, filename, b64_data):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return path


def save_unique_face_snaps(run_dir, unique_faces):
    faces_dir = os.path.join(run_dir, "faces")
    os.makedirs(faces_dir, exist_ok=True)
    for face in unique_faces:
        b64 = face.get("snap_base64")
        if not b64:
            continue
        name = face.get("name") or face["person_id"]
        safe_name = safe_filename(name)
        filename = f"{safe_name}.png"
        path = os.path.join(faces_dir, filename)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        face["snap_filename"] = filename
        face["snap_path"] = path


def save_unique_object_snaps(run_dir, unique_objects):
    objects_dir = os.path.join(run_dir, "objects")
    os.makedirs(objects_dir, exist_ok=True)
    for obj in unique_objects:
        b64 = obj.get("snap_base64")
        if not b64:
            continue
        oid = obj["object_id"]
        ident = safe_filename(obj.get("identification", "object"))
        filename = f"{oid}_{ident}.jpg"
        path = os.path.join(objects_dir, filename)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        obj["snap_filename"] = filename
        obj["snap_path"] = path


def get_output_path(filename):
    return os.path.join(OUTPUT_DIR, filename)


def json_safe(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            str(key): json_safe(inner)
            for key, inner in value.items()
            if not callable(inner)
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return str(value)


def prepare_detection_metadata(records):
    prepared = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        prepared.append(json_safe({
            key: value
            for key, value in record.items()
            if key != "snap_base64"
        }))
    return prepared


def save_detection_metadata(run_dir, unique_faces, unique_objects):
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "version": 1,
        "unique_faces": prepare_detection_metadata(unique_faces),
        "unique_objects": prepare_detection_metadata(unique_objects),
    }
    path = os.path.join(run_dir, DETECTION_METADATA_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return path


def load_detection_metadata(run_dir):
    path = os.path.join(run_dir, DETECTION_METADATA_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def load_snap_base64_map(directory):
    if not os.path.isdir(directory):
        return {}

    snapshots = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(directory, fn)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                snapshots[fn] = base64.b64encode(f.read()).decode("ascii")
        except Exception:
            continue
    return snapshots


def _find_snapshot_by_candidates(snapshots_by_filename, candidates):
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in snapshots_by_filename:
            return candidate

    normalized_candidates = []
    for candidate in candidates:
        if not candidate:
            continue
        base, ext = os.path.splitext(candidate)
        if ext:
            normalized_candidates.append(candidate)
        else:
            normalized_candidates.extend([f"{candidate}.png", f"{candidate}.jpg", f"{candidate}.jpeg"])

    for candidate in normalized_candidates:
        if candidate in snapshots_by_filename:
            return candidate
    return None


def infer_face_snapshot_name(item, snapshots_by_filename):
    candidates = []
    person_id = str(item.get("person_id") or "").strip()
    name = str(item.get("name") or "").strip()
    if person_id:
        candidates.append(safe_filename(person_id))
    if name:
        candidates.append(safe_filename(name))

    match = _find_snapshot_by_candidates(snapshots_by_filename, candidates)
    if match:
        return match

    for filename in sorted(snapshots_by_filename):
        stem, _ = os.path.splitext(filename)
        if stem == person_id or stem == safe_filename(person_id) or stem == safe_filename(name):
            return filename
    return None


def infer_object_snapshot_name(item, snapshots_by_filename):
    object_id = str(item.get("object_id") or "").strip()
    identification = str(item.get("identification") or "").strip()
    safe_identification = safe_filename(identification or "object")

    candidates = []
    if object_id and safe_identification:
        candidates.append(f"{object_id}_{safe_identification}")
    if object_id:
        candidates.append(object_id)

    match = _find_snapshot_by_candidates(snapshots_by_filename, candidates)
    if match:
        return match

    if object_id:
        prefix = f"{object_id}_"
        for filename in sorted(snapshots_by_filename):
            if filename.startswith(prefix):
                return filename
    return None


def attach_snapshots(records, snapshots_by_filename, infer_filename=None):
    enriched = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        item = dict(record)
        snapshot_name = item.get("snap_filename")
        if not snapshot_name and item.get("snap_path"):
            snapshot_name = os.path.basename(str(item["snap_path"]))
        if not snapshot_name and infer_filename is not None:
            snapshot_name = infer_filename(item, snapshots_by_filename)
            if snapshot_name and "snap_filename" not in item:
                item["snap_filename"] = snapshot_name
        if snapshot_name and snapshot_name in snapshots_by_filename:
            item["snap_base64"] = snapshots_by_filename[snapshot_name]
        enriched.append(item)
    return enriched


def load_faces_objects_from_disk(job_id):
    """
    Load unique_faces and unique_objects from snaps/<job_id>/faces and snaps/<job_id>/objects.
    Returns None if the run dir does not exist; otherwise {"unique_faces": [...], "unique_objects": [...]}.
    Image filenames: faces = person_N.png; objects = object_N_<name>.jpg (name from filename).
    """
    run_dir = os.path.join(SNAPS_DIR, job_id)
    if not os.path.isdir(run_dir):
        return None
    faces_dir = os.path.join(run_dir, "faces")
    objects_dir = os.path.join(run_dir, "objects")
    metadata = load_detection_metadata(run_dir)
    if metadata is not None:
        return {
            "unique_faces": attach_snapshots(
                metadata.get("unique_faces", []),
                load_snap_base64_map(faces_dir),
                infer_filename=infer_face_snapshot_name,
            ),
            "unique_objects": attach_snapshots(
                metadata.get("unique_objects", []),
                load_snap_base64_map(objects_dir),
                infer_filename=infer_object_snapshot_name,
            ),
        }

    unique_faces = []
    unique_objects = []

    if os.path.isdir(faces_dir):
        for fn in sorted(os.listdir(faces_dir)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(faces_dir, fn)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
            except Exception:
                continue
            name = os.path.splitext(fn)[0]
            unique_faces.append({
                "person_id": name,
                "snap_base64": b64,
                "description": name.replace("_", " ").title(),
            })

    if os.path.isdir(objects_dir):
        for fn in sorted(os.listdir(objects_dir)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(objects_dir, fn)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
            except Exception:
                continue
            base = os.path.splitext(fn)[0]
            parts = base.split("_")
            oid = parts[0] + "_" + parts[1] if len(parts) >= 2 else base
            identification = "_".join(parts[2:]) if len(parts) > 2 else (parts[1] if len(parts) >= 2 else base)
            unique_objects.append({
                "object_id": oid,
                "identification": identification.replace("_", " ").title(),
                "snap_base64": b64,
            })

    return {"unique_faces": unique_faces, "unique_objects": unique_objects}
