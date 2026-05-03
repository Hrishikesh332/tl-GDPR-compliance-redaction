import copy
import json
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.detection import detect_face_boxes, localize_known_faces_in_frame
from services.face_identity import get_face_identity
from services.pipeline import get_enriched_faces, get_job
from services.redactor import (
    expand_face_redaction_bbox,
    filter_reverse_focus_detected_tracks,
    redact_video,
)


JOB_ID = "e4497c33-7a4"
WORKDIR = Path("/private/tmp/video-redaction-reverse-render-tests")
CASES = [
    {
        "name": "t407_no_focus_blur_all",
        "start": 406.0,
        "duration": 2.0,
        "focus": None,
        "check_sec": 1.0,
        "expected_min_blurs": 6,
    },
    {
        "name": "t30_focus_person_2",
        "start": 29.0,
        "duration": 2.0,
        "focus": "person_2",
        "check_sec": 1.0,
        "expected_min_blurs": 3,
    },
    {
        "name": "t295_focus_person_2",
        "start": 294.0,
        "duration": 2.0,
        "focus": "person_2",
        "check_sec": 1.0,
        "expected_min_blurs": 4,
    },
    {
        "name": "t407_focus_person_11",
        "start": 406.0,
        "duration": 2.0,
        "focus": "person_11",
        "check_sec": 1.0,
        "expected_min_blurs": 5,
    },
]


def scaled_bbox(bbox, scale_x, scale_y):
    if not bbox:
        return bbox
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return [
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    ]


def shifted_face(face, start, scale_x, scale_y):
    shifted = copy.deepcopy(face)
    if shifted.get("bbox"):
        shifted["bbox"] = scaled_bbox(shifted["bbox"], scale_x, scale_y)
    for appearance in shifted.get("appearances") or []:
        if "timestamp" in appearance:
            appearance["timestamp"] = float(appearance["timestamp"]) - start
        if "frame_idx" in appearance:
            appearance["frame_idx"] = max(0, int(round(float(appearance.get("timestamp", 0.0)) * 25)))
        if appearance.get("bbox"):
            appearance["bbox"] = scaled_bbox(appearance["bbox"], scale_x, scale_y)
    shifted["time_ranges"] = [
        {
            "start_sec": max(0.0, float(time_range.get("start_sec", time_range.get("start", 0.0))) - start),
            "end_sec": max(0.0, float(time_range.get("end_sec", time_range.get("end", 0.0))) - start),
        }
        for time_range in (shifted.get("time_ranges") or [])
    ]
    return shifted


def mean_abs_delta(a, b):
    return float(abs(a.astype("float32") - b.astype("float32")).mean())


def frame_at(video_path, seconds):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(seconds * fps)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at {seconds}s from {video_path}")
    return frame


def clip_source(source, clip_path, start, duration):
    subprocess.run(
        [
            "/opt/homebrew/bin/ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            str(start),
            "-i",
            source,
            "-t",
            str(duration),
            "-vf",
            "scale=-2:360",
            "-an",
            str(clip_path),
        ],
        check=True,
    )


def main():
    job = get_job(JOB_ID)
    if not job:
        raise RuntimeError(f"Missing job {JOB_ID}")

    faces = (get_enriched_faces(JOB_ID) or {}).get("unique_faces") or job.get("unique_faces") or []
    faces_by_id = {get_face_identity(face): face for face in faces if get_face_identity(face)}
    WORKDIR.mkdir(parents=True, exist_ok=True)

    results = []
    for case in CASES:
        clip_path = WORKDIR / f"{case['name']}_clip.mp4"
        output_path = WORKDIR / f"{case['name']}_redacted.mp4"
        clip_source(job["video_path"], clip_path, case["start"], case["duration"])
        source_frame = frame_at(clip_path, case["check_sec"])
        frame_h, frame_w = source_frame.shape[:2]
        source_cap = cv2.VideoCapture(job["video_path"])
        source_w = source_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame_w
        source_h = source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame_h
        source_cap.release()
        focus_face = None
        if case.get("focus"):
            focus_face = shifted_face(
                faces_by_id[case["focus"]],
                case["start"],
                frame_w / source_w,
                frame_h / source_h,
            )

        render_result = redact_video(
            input_path=str(clip_path),
            output_path=str(output_path),
            face_encodings=["__ALL__"],
            face_targets=[],
            object_classes=set(),
            detect_every_n=1,
            output_height=360,
            reverse_face_redaction=True,
            preserve_face_targets=[focus_face] if focus_face else [],
            blur_strength=220,
        )

        redacted_frame = frame_at(output_path, case["check_sec"])

        preserve_bboxes = []
        if focus_face:
            preserve_bboxes = [
                expand_face_redaction_bbox(tuple(face["bbox"]), frame_w, frame_h)
                for face in localize_known_faces_in_frame(
                    source_frame,
                    [focus_face],
                    time_sec=case["check_sec"],
                    tolerance=0.55,
                )
            ]
        baseline_tracks = [
            {"kind": "face", "bbox": expand_face_redaction_bbox(tuple(face["bbox"]), frame_w, frame_h)}
            for face in detect_face_boxes(
                source_frame,
                confidence_threshold=0.16,
                include_supplemental=True,
                min_face_size=10,
                min_sharpness=2.0,
                upscale=2.2,
            )
        ]
        blur_targets = filter_reverse_focus_detected_tracks(list(baseline_tracks), preserve_bboxes)

        blur_checks = []
        for idx, track in enumerate(blur_targets):
            x1, y1, x2, y2 = [int(round(v)) for v in track["bbox"]]
            x1 = max(0, min(x1, frame_w - 1))
            x2 = max(x1 + 1, min(x2, frame_w))
            y1 = max(0, min(y1, frame_h - 1))
            y2 = max(y1 + 1, min(y2, frame_h))
            source_crop = source_frame[y1:y2, x1:x2]
            redacted_crop = redacted_frame[y1:y2, x1:x2]
            target_delta = mean_abs_delta(source_crop, redacted_crop)
            ctrl_y1 = min(max(0, y1 + 80), frame_h - (y2 - y1))
            ctrl_y2 = ctrl_y1 + (y2 - y1)
            control_delta = mean_abs_delta(
                source_frame[ctrl_y1:ctrl_y2, x1:x2],
                redacted_frame[ctrl_y1:ctrl_y2, x1:x2],
            )
            blur_checks.append(
                {
                    "idx": idx,
                    "target_delta": round(target_delta, 2),
                    "control_delta": round(control_delta, 2),
                    "ratio": round(target_delta / max(control_delta, 1e-6), 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        focus_checks = []
        for idx, bbox in enumerate(preserve_bboxes):
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1 = max(0, min(x1, frame_w - 1))
            x2 = max(x1 + 1, min(x2, frame_w))
            y1 = max(0, min(y1, frame_h - 1))
            y2 = max(y1 + 1, min(y2, frame_h))
            source_crop = source_frame[y1:y2, x1:x2]
            redacted_crop = redacted_frame[y1:y2, x1:x2]
            target_delta = mean_abs_delta(source_crop, redacted_crop)
            ctrl_y1 = min(max(0, y1 + 80), frame_h - (y2 - y1))
            ctrl_y2 = ctrl_y1 + (y2 - y1)
            control_delta = mean_abs_delta(
                source_frame[ctrl_y1:ctrl_y2, x1:x2],
                redacted_frame[ctrl_y1:ctrl_y2, x1:x2],
            )
            focus_checks.append(
                {
                    "idx": idx,
                    "target_delta": round(target_delta, 2),
                    "control_delta": round(control_delta, 2),
                    "ratio": round(target_delta / max(control_delta, 1e-6), 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        blurred_ok = sum(
            1 for item in blur_checks
            if item["target_delta"] >= max(6.0, item["control_delta"] * 1.25)
        )
        focus_ok = all(
            item["target_delta"] <= max(24.0, item["control_delta"] * 2.8)
            for item in focus_checks
        ) if focus_checks else True
        case_result = {
            "case": case["name"],
            "output": str(output_path),
            "frames_processed": render_result.get("frames_processed"),
            "baseline_faces": len(baseline_tracks),
            "focus_matches": len(preserve_bboxes),
            "blur_targets": len(blur_targets),
            "blurred_ok": blurred_ok,
            "expected_min_blurs": case["expected_min_blurs"],
            "focus_ok": focus_ok,
            "blur_checks": blur_checks,
            "focus_checks": focus_checks,
        }
        results.append(case_result)
        print(json.dumps(case_result, sort_keys=True), flush=True)

    failures = []
    for result in results:
        if result["blur_targets"] < result["expected_min_blurs"]:
            failures.append({"case": result["case"], "reason": "not enough blur targets", "result": result})
        if result["blurred_ok"] < result["expected_min_blurs"]:
            failures.append({"case": result["case"], "reason": "not enough targets visibly blurred", "result": result})
        if not result["focus_ok"]:
            failures.append({"case": result["case"], "reason": "focus face appears over-blurred", "result": result})

    print("RENDER_TEST_SUMMARY")
    print(json.dumps({"cases": len(results), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)
    print("ALL_RENDER_TESTS_PASSED")


if __name__ == "__main__":
    main()
