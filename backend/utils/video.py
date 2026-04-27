import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np

logger = logging.getLogger("video_redaction.video")

ANALYSIS_SAMPLE_INTERVAL = 5.0
EXPORT_HEIGHTS = (480, 720, 1080)


def normalize_export_height(value, default=720):
    """Normalize requested export quality to one of the supported heights."""
    if value is None or value == "":
        return default
    try:
        if isinstance(value, str):
            raw = value.strip().lower().removesuffix("p")
            height = int(raw)
        else:
            height = int(value)
    except (TypeError, ValueError):
        return default
    return height if height in EXPORT_HEIGHTS else default


def even_dimension(value):
    dimension = int(round(float(value)))
    if dimension % 2:
        dimension -= 1
    return max(2, dimension)


def export_video_dimensions(source_width, source_height, target_height):
    """Return even output dimensions, preserving aspect ratio and avoiding upscaling."""
    source_width = int(source_width or 0)
    source_height = int(source_height or 0)
    target_height = normalize_export_height(target_height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Cannot compute export dimensions for invalid source video size")

    output_height = source_height if source_height <= target_height else target_height
    output_height = even_dimension(output_height)
    scale = output_height / float(source_height)
    output_width = even_dimension(source_width * scale)
    return output_width, output_height


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_sec": round(duration, 2),
    }


def extract_keyframes(video_path, interval_sec=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval_frames = max(1, int(fps * interval_sec))
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval_frames == 0:
            timestamp = round(frame_idx / fps, 3)
            frames.append({
                "frame": frame,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
            })
        frame_idx += 1

    cap.release()
    return frames


def extract_frames_at_timestamps(video_path, timestamps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    for ts in sorted(timestamps):
        target_frame = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret:
            frames.append({
                "frame": frame,
                "frame_idx": target_frame,
                "timestamp": round(ts, 3),
            })

    cap.release()
    return frames


def extract_frame_at_time(video_path, time_sec):
    """Extract the closest frame for a given timestamp in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    safe_time = max(0.0, float(time_sec or 0.0))
    if total_frames > 0:
        target_frame = max(0, min(total_frames - 1, int(round(safe_time * fps))))
    else:
        target_frame = int(round(safe_time * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret and target_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - 1))
        ret, frame = cap.read()
        target_frame = max(0, target_frame - 1)

    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not read frame at {safe_time:.3f}s from {video_path}")

    timestamp = (target_frame / fps) if fps > 0 else safe_time
    return {
        "frame": frame,
        "frame_idx": target_frame,
        "timestamp": round(timestamp, 4),
        "fps": fps,
    }


def merge_overlapping_ranges(ranges):
    """Merge overlapping/adjacent time ranges into non-overlapping intervals."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1] + 0.5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def timestamps_from_time_ranges(time_ranges, sample_interval=None):
    """Convert time ranges into smart keyframe timestamps.

    Instead of sampling every second (which produces as many frames as uniform
    sampling), this function:
    1. Merges overlapping ranges
    2. Samples at ANALYSIS_SAMPLE_INTERVAL (default 5s) within each range
    3. Always includes start and end boundaries of each range
    """
    if sample_interval is None:
        sample_interval = ANALYSIS_SAMPLE_INTERVAL

    raw_ranges = []
    for tr in time_ranges:
        start = tr.get("start_sec") or tr.get("start") or 0
        end = tr.get("end_sec") or tr.get("end") or 0
        if end < start:
            end = start
        raw_ranges.append((float(start), float(end)))

    if not raw_ranges:
        return []

    merged = merge_overlapping_ranges(raw_ranges)

    total_coverage = sum(e - s for s, e in merged)
    logger.info("Analysis ranges: %d raw -> %d merged, covering %.1fs of video (sample_interval=%.1fs)",
                len(raw_ranges), len(merged), total_coverage, sample_interval)

    ts_set = set()
    for start, end in merged:
        ts_set.add(round(start, 3))
        ts_set.add(round(end, 3))
        if end - start > sample_interval:
            t = start + sample_interval
            while t < end:
                ts_set.add(round(t, 3))
                t += sample_interval

    result = sorted(ts_set)
    logger.info("Generated %d keyframe timestamps from analysis (was %d raw ranges)",
                len(result), len(raw_ranges))
    return result


def reencode_mp4_to_h264(input_path, output_path=None, original_path=None):
    """Re-encode an MP4  to H.264 for universal playback.

    If *original_path* is provided, the audio stream from that file is muxed
    into the output so the redacted video keeps its soundtrack.
    """
    if output_path is None:
        output_path = input_path
    if input_path == output_path:
        fd, temp_out = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(output_path) or ".")
        os.close(fd)
        final_path = temp_out
    else:
        final_path = output_path

    def promote_input_to_output(reason):
        if input_path == output_path:
            exists = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
            if exists:
                logger.warning("%s; keeping OpenCV-generated MP4: %s", reason, output_path)
            return False

        try:
            if os.path.isfile(output_path):
                os.remove(output_path)
            os.replace(input_path, output_path)
            logger.warning("%s; using OpenCV-generated MP4 fallback: %s", reason, output_path)
            return False
        except Exception as exc:
            logger.warning("Could not promote OpenCV MP4 fallback to %s: %s", output_path, exc)
            return False

    try:
        cmd = ["ffmpeg", "-y", "-i", input_path]
        if original_path and os.path.isfile(original_path):
            cmd += ["-i", original_path]
        cmd += [
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
        ]
        if original_path and os.path.isfile(original_path):
            cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-shortest"]
        cmd.append(final_path)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.warning("ffmpeg re-encode failed: %s", result.stderr[-500:] if result.stderr else "unknown")
            if final_path != input_path and os.path.isfile(final_path):
                try:
                    os.remove(final_path)
                except OSError:
                    pass
            return promote_input_to_output("ffmpeg re-encode failed")
        if final_path != output_path:
            os.replace(final_path, output_path)
        logger.info("Re-encoded to H.264: %s", output_path)
        return True
    except FileNotFoundError:
        return promote_input_to_output("ffmpeg not found; output remains mp4v")
    except Exception as e:
        logger.warning("Re-encode failed: %s", e)
        if final_path != input_path and os.path.isfile(final_path):
            try:
                os.remove(final_path)
            except OSError:
                pass
        return promote_input_to_output("Re-encode failed")


def validate_mp4_output(video_path):
    """Ensure a finished export is a readable, non-empty MP4 file."""
    if not video_path or not str(video_path).lower().endswith(".mp4"):
        raise ValueError(f"Redacted export is not an MP4 path: {video_path}")
    if not os.path.isfile(video_path):
        raise ValueError(f"Redacted MP4 was not created: {video_path}")
    size_bytes = os.path.getsize(video_path)
    if size_bytes <= 0:
        raise ValueError(f"Redacted MP4 is empty: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Redacted MP4 cannot be opened: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, _frame = cap.read()
    cap.release()

    if width <= 0 or height <= 0 or not ret:
        raise ValueError(f"Redacted MP4 has no readable video frames: {video_path}")
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": max(total_frames, 1),
        "size_bytes": size_bytes,
    }


def finalize_mp4_export(rendered_path, output_path, original_path=None):
    """Finalize a rendered redaction file into a validated downloadable MP4."""
    if not rendered_path or not os.path.isfile(rendered_path):
        raise ValueError(f"Rendered MP4 was not created: {rendered_path}")
    if not output_path or not str(output_path).lower().endswith(".mp4"):
        raise ValueError(f"Output path must be an MP4: {output_path}")

    rendered_real = os.path.realpath(rendered_path)
    output_real = os.path.realpath(output_path)
    h264_encoded = reencode_mp4_to_h264(rendered_path, output_path, original_path=original_path)
    try:
        metadata = validate_mp4_output(output_path)
    except Exception:
        if rendered_real == output_real or not os.path.isfile(rendered_path):
            raise
        logger.warning(
            "Final H.264 output was not readable; promoting OpenCV MP4 fallback: %s",
            output_path,
            exc_info=True,
        )
        if os.path.isfile(output_path):
            os.remove(output_path)
        os.replace(rendered_path, output_path)
        h264_encoded = False
        metadata = validate_mp4_output(output_path)
    finally:
        try:
            if rendered_real != output_real and os.path.isfile(rendered_path):
                os.remove(rendered_path)
        except OSError:
            logger.debug("Could not remove temporary rendered MP4: %s", rendered_path, exc_info=True)

    metadata["h264_encoded"] = h264_encoded
    return metadata


def small_frame_for_tracking(frame, max_dim=480):
    h, w = frame.shape[:2]
    if max(w, h) <= max_dim:
        return frame, 1.0
    scale = max_dim / max(w, h)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return small, 1.0 / scale
