"""Package and install only the repository's explicitly catalogued demo jobs."""
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

logger = logging.getLogger("video_redaction.demo_data")


def read_catalog(backend):
    return json.loads((Path(backend) / "demo" / "catalog.json").read_text())


def demo_manifest(backend, entry):
    backend = Path(backend)
    for key in ("job_id", "source_file"):
        value = entry[key]
        if not value or Path(value).name != value or value in (".", ".."):
            raise ValueError(f"Invalid demo {key}")
    folder = backend / "snaps" / entry["job_id"]
    manifest = json.loads((folder / "job_manifest.json").read_text())
    if manifest.get("twelvelabs_video_id") != entry["video_id"]:
        raise ValueError(f"Demo video/job mismatch: {entry['job_id']}")
    metadata = json.loads((folder / "detection_metadata.json").read_text())
    if not metadata.get("unique_faces"):
        raise ValueError(f"Demo has no saved face detections: {entry['job_id']}")
    for face in metadata["unique_faces"]:
        name = face.get("snap_filename") or Path(face.get("snap_path", "")).name
        if not name:
            name = f"{face['person_id']}.png"
        if Path(name).name != name or not (folder / "faces" / name).is_file():
            raise ValueError(f"Demo face snapshot missing: {entry['job_id']}/{name}")
    return manifest


def verify_video(path, entry):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != entry["source_size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == entry["source_sha256"]


def prepare_demo_assets(backend):
    """Resolve LFS at build time from a pinned repo revision, then check hashes."""
    import requests

    backend = Path(backend)
    catalog = read_catalog(backend)
    for entry in catalog["videos"]:
        demo_manifest(backend, entry)
        path = backend / "output" / entry["source_file"]
        if verify_video(path, entry):
            continue
        # The catalogue pins both the repository commit and Git LFS SHA-256.
        url = (
            f"https://media.githubusercontent.com/media/{catalog['repository']}/"
            f"{catalog['source_revision']}/backend/output/{entry['source_file']}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".download")
        try:
            with os.fdopen(fd, "wb") as target:
                with requests.get(url, stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(1024 * 1024):
                        target.write(chunk)
            if not verify_video(tmp, entry):
                raise ValueError(f"Demo source checksum mismatch: {entry['source_file']}")
            os.replace(tmp, path)
        finally:
            Path(tmp).unlink(missing_ok=True)
        print(f"Verified demo source: {entry['source_file']}", flush=True)
    print(f"Verified {len(catalog['videos'])} demo video/detection bundles", flush=True)


def seed_demo_data(backend, data_dir):
    """Copy missing demo metadata to the volume; never replace user job state."""
    backend, data_dir = Path(backend).resolve(), Path(data_dir).resolve()
    if backend == data_dir:
        return 0  # Local checkouts already contain the saved jobs.
    snaps = data_dir / "snaps"
    snaps.mkdir(parents=True, exist_ok=True)
    bound_videos = set()
    for path in snaps.glob("*/job_manifest.json"):
        try:
            manifest = json.loads(path.read_text())
            bound_videos.add(manifest.get("twelvelabs_video_id"))
        except (ValueError, OSError):
            continue
    installed = 0
    for entry in read_catalog(backend)["videos"]:
        destination = snaps / entry["job_id"]
        if destination.exists() or entry["video_id"] in bound_videos:
            continue
        manifest = demo_manifest(backend, entry)
        source_video = backend / "output" / entry["source_file"]
        if not source_video.is_file() or source_video.stat().st_size != entry["source_size"]:
            raise ValueError(f"Demo source is not packaged: {entry['source_file']}")
        # Demo sources are immutable image assets. New uploads continue to use
        # DATA_DIR/output; this exact path avoids guessing another video's file.
        manifest.update({
            "video_path": str(source_video),
            "demo_seed_version": 1,
            "status": "ready",
            "local_status": "done",
        })
        temporary = Path(tempfile.mkdtemp(prefix=".demo-seed-", dir=snaps))
        try:
            shutil.copytree(backend / "snaps" / entry["job_id"], temporary, dirs_exist_ok=True)
            (temporary / "job_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        bound_videos.add(entry["video_id"])
        installed += 1
    logger.info("Installed %d missing demo detection jobs; existing jobs preserved", installed)
    return installed
