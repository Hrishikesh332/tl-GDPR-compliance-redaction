import base64
import os
import re

from config import SNAPS_DIR, OUTPUT_DIR


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
        path = os.path.join(faces_dir, f"{safe_name}.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
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
        path = os.path.join(objects_dir, f"{oid}_{ident}.jpg")
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        obj["snap_path"] = path


def get_output_path(filename):
    return os.path.join(OUTPUT_DIR, filename)


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
