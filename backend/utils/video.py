import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np

logger = logging.getLogger("video_redaction.video")

ANALYSIS_SAMPLE_INTERVAL = 5.0


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


def _merge_overlapping_ranges(ranges):
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

    merged = _merge_overlapping_ranges(raw_ranges)

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


def reencode_mp4_to_h264(input_path, output_path=None):
    """Re-encode an MP4 (e.g. mp4v) to H.264 for universal playback (QuickTime, browsers, etc.)."""
    if output_path is None:
        output_path = input_path
    if input_path == output_path:
        fd, temp_out = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(output_path) or ".")
        os.close(fd)
        final_path = temp_out
    else:
        final_path = output_path

    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            final_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.warning("ffmpeg re-encode failed: %s", result.stderr[-500:] if result.stderr else "unknown")
            return False
        if final_path != output_path:
            os.replace(final_path, output_path)
        logger.info("Re-encoded to H.264: %s", output_path)
        return True
    except FileNotFoundError:
        logger.warning("ffmpeg not found; output remains mp4v (may not play in QuickTime)")
        return False
    except Exception as e:
        logger.warning("Re-encode failed: %s", e)
        return False


def small_frame_for_tracking(frame, max_dim=480):
    h, w = frame.shape[:2]
    if max(w, h) <= max_dim:
        return frame, 1.0
    scale = max_dim / max(w, h)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return small, 1.0 / scale
