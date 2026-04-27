import os
import re


REDACTED_MP4_PATTERN = re.compile(
    r"^redacted_[A-Za-z0-9][A-Za-z0-9_-]{2,80}_\d{8}_\d{6}(?:_\d{3,6})?(?:_(?:480|720|1080)p)?\.mp4$",
    re.IGNORECASE,
)


def safe_redacted_mp4_filename(filename):
    """Return a trusted redacted export filename or raise ValueError."""
    candidate = os.path.basename(str(filename or "").strip())
    if not candidate or candidate != filename:
        raise ValueError("invalid download filename")
    if not REDACTED_MP4_PATTERN.fullmatch(candidate):
        raise ValueError("download must be a redacted MP4 export")
    return candidate


def redacted_download_path(filename, output_dir):
    """Resolve a redacted export path while keeping it inside output_dir."""
    safe = safe_redacted_mp4_filename(filename)
    root = os.path.realpath(output_dir)
    path = os.path.realpath(os.path.join(root, safe))
    if os.path.commonpath([root, path]) != root:
        raise ValueError("download path escapes output directory")
    return safe, path
