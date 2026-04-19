#!/usr/bin/env python3
"""Generate midpoint thumbnails and persist their URLs to TwelveLabs metadata.

This script creates custom poster images from local source videos, stores them in
``frontend/public/generated-thumbnails``, and writes the resulting relative URL
to ``user_metadata.preferred_thumbnail_url`` for each TwelveLabs video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parents[1]
repo_dir = backend_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config import TWELVELABS_INDEX_ID
from services import twelvelabs_service
from services.pipeline import infer_video_path_for_video
from services.twelvelabs_service_helpers import PREFERRED_THUMBNAIL_URL_KEY
from utils.video import extract_frame_at_time, get_video_metadata

DEFAULT_VIDEO_IDS = [
    "69b67a2968f910812bf38503",
    "69b67a433571b38304a513f1",
    "69b67a843571b38304a513f9",
]
DEFAULT_OUTPUT_DIR = repo_dir / "frontend" / "public" / "generated-thumbnails"
DEFAULT_URL_PREFIX = "/generated-thumbnails"
PREFERRED_THUMBNAIL_INDEX_KEY = "preferred_thumbnail_index"
PREFERRED_THUMBNAIL_STRATEGY_KEY = "preferred_thumbnail_strategy"
PREFERRED_THUMBNAIL_UPDATED_BY_KEY = "preferred_thumbnail_updated_by"
PREFERRED_THUMBNAIL_TIME_KEY = "preferred_thumbnail_time_sec"
MIDPOINT_STRATEGY = "generated_midpoint_frame"
SCRIPT_MARKER = "backend/scripts/update_twelvelabs_midpoint_thumbnails.py"
FRAME_FRACTIONS = (0.50, 0.45, 0.55, 0.40, 0.60, 0.35, 0.65, 0.30, 0.70)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate midpoint thumbnails and store their URLs in TwelveLabs user metadata."
    )
    parser.add_argument(
        "video_ids",
        nargs="*",
        default=DEFAULT_VIDEO_IDS,
        help="TwelveLabs video IDs to update. Defaults to the requested three IDs.",
    )
    parser.add_argument(
        "--index-id",
        default=TWELVELABS_INDEX_ID,
        help="TwelveLabs index ID. Defaults to TWELVELABS_INDEX_ID from backend/.env.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where generated thumbnail images should be written.",
    )
    parser.add_argument(
        "--url-prefix",
        default=DEFAULT_URL_PREFIX,
        help="Public URL prefix for generated thumbnails. Stored in TwelveLabs metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the intended updates without writing thumbnails or patching TwelveLabs.",
    )
    return parser.parse_args()


def normalize_url_prefix(value):
    text = str(value or "").strip() or DEFAULT_URL_PREFIX
    if not text.startswith("/"):
        text = f"/{text}"
    return text.rstrip("/")


def thumbnail_url_for_video(video_id, url_prefix):
    return f"{normalize_url_prefix(url_prefix)}/{video_id}.jpg"


def frame_visual_score(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    std_luma = float(np.std(gray))
    return mean_luma + (std_luma * 0.65), mean_luma, std_luma


def choose_thumbnail_frame(video_path, duration_sec):
    if duration_sec <= 0:
        raise ValueError(f"Video duration is invalid for thumbnail generation: {video_path}")

    best_candidate = None
    for index, fraction in enumerate(FRAME_FRACTIONS):
        time_sec = max(0.0, min(duration_sec, duration_sec * fraction))
        frame_info = extract_frame_at_time(video_path, time_sec)
        frame = frame_info["frame"]
        score, mean_luma, std_luma = frame_visual_score(frame)
        candidate = {
            "frame": frame,
            "fraction": fraction,
            "time_sec": float(frame_info["timestamp"]),
            "score": score - (index * 0.8),
            "mean_luma": mean_luma,
            "std_luma": std_luma,
        }

        if mean_luma >= 28.0 and std_luma >= 10.0:
            return candidate

        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate

    if best_candidate is None:
        raise ValueError(f"Could not extract any usable frame from {video_path}")
    return best_candidate


def write_thumbnail_image(frame_bgr, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        raise RuntimeError(f"Could not write thumbnail image to {output_path}")


def resolve_local_video_path(video_id, index_id):
    info = twelvelabs_service.get_video_info(video_id, index_id=index_id)
    video_path = infer_video_path_for_video(video_id, info=info)
    if not video_path:
        raise FileNotFoundError(f"No local source video could be inferred for TwelveLabs video {video_id}")
    return info, Path(video_path)


def apply_thumbnail_override(video_id, index_id, output_dir, url_prefix, dry_run=False):
    info, video_path = resolve_local_video_path(video_id, index_id=index_id)
    if not video_path.is_file():
        raise FileNotFoundError(f"Local source video does not exist: {video_path}")

    metadata = info.get("system_metadata") or {}
    duration_sec = float(metadata.get("duration") or 0.0)
    if duration_sec <= 0:
        duration_sec = float(get_video_metadata(str(video_path)).get("duration_sec") or 0.0)
    if duration_sec <= 0:
        raise ValueError(f"Could not determine video duration for {video_id}")

    selected_frame = choose_thumbnail_frame(str(video_path), duration_sec)
    relative_thumbnail_url = thumbnail_url_for_video(video_id, url_prefix)
    output_path = Path(output_dir) / f"{video_id}.jpg"
    current_metadata = dict(info.get("user_metadata") or {})
    current_thumbnail_url = current_metadata.get(PREFERRED_THUMBNAIL_URL_KEY)

    patch_payload = {
        PREFERRED_THUMBNAIL_URL_KEY: relative_thumbnail_url,
        PREFERRED_THUMBNAIL_INDEX_KEY: int(round(selected_frame["fraction"] * 100)),
        PREFERRED_THUMBNAIL_STRATEGY_KEY: MIDPOINT_STRATEGY,
        PREFERRED_THUMBNAIL_UPDATED_BY_KEY: SCRIPT_MARKER,
        PREFERRED_THUMBNAIL_TIME_KEY: round(selected_frame["time_sec"], 3),
    }

    unchanged = (
        current_thumbnail_url == relative_thumbnail_url
        and current_metadata.get(PREFERRED_THUMBNAIL_STRATEGY_KEY) == MIDPOINT_STRATEGY
        and output_path.is_file()
    )
    if unchanged:
        return {
            "video_id": video_id,
            "status": "unchanged",
            "video_path": str(video_path),
            "output_path": str(output_path),
            "preferred_thumbnail_url": relative_thumbnail_url,
            "selected_time_sec": round(selected_frame["time_sec"], 3),
        }

    if not dry_run:
        write_thumbnail_image(selected_frame["frame"], output_path)
        twelvelabs_service.twelvelabs_api_request(
            "PATCH",
            f"indexes/{index_id}/videos/{video_id}",
            json_body={"user_metadata": patch_payload},
            expected_status=(200, 204),
        )

    return {
        "video_id": video_id,
        "status": "dry_run" if dry_run else "updated",
        "video_path": str(video_path),
        "output_path": str(output_path),
        "preferred_thumbnail_url": relative_thumbnail_url,
        "previous_thumbnail_url": current_thumbnail_url,
        "selected_time_sec": round(selected_frame["time_sec"], 3),
        "selected_fraction": selected_frame["fraction"],
        "mean_luma": round(selected_frame["mean_luma"], 2),
        "std_luma": round(selected_frame["std_luma"], 2),
        "metadata_patch": patch_payload,
    }


def main():
    arguments = parse_arguments()
    output_dir = Path(arguments.output_dir)
    results = []

    for raw_video_id in arguments.video_ids:
        video_id = str(raw_video_id).strip()
        if not video_id:
            continue
        try:
            result = apply_thumbnail_override(
                video_id,
                index_id=arguments.index_id,
                output_dir=output_dir,
                url_prefix=arguments.url_prefix,
                dry_run=arguments.dry_run,
            )
        except Exception as error:
            result = {
                "video_id": video_id,
                "status": "error",
                "error": str(error),
            }
        results.append(result)
        print(json.dumps(result, indent=2))

    if any(item.get("status") == "error" for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
