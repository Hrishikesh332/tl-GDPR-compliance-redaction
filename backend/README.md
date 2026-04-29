# Backend API


- Default local URL: `http://localhost:5001`
- API prefix: `/api`
- Main entry point: `app.py`
- Route registration: `routes/__init__.py`
- Runtime log: `logs/pipeline.log`

## What The Backend Does

The backend supports two related workflows:

1. TwelveLabs indexing and search
   - Upload or register videos with TwelveLabs.
   - List indexed videos and retrieve HLS playback metadata.
   - Run text, image, and entity-assisted search.
   - Store lightweight video overview metadata.

2. Local analysis and redaction
   - Save uploaded source videos in `output/`.
   - Extract keyframes and run face/object detection.
   - Cluster detections into unique people and object classes.
   - Save detection manifests and snapshots in `snaps/<job_id>/`.
   - Preview live redaction boxes and tracked custom regions.
   - Render redacted MP4 exports with blur or black-box redaction.

## Runtime Stack

- Flask + Flask-CORS
- TwelveLabs Python SDK, with REST fallbacks for entity endpoints
- OpenCV for video I/O, face detection, tracking, and frame processing
- InsightFace + ONNX Runtime for face embeddings when available
- Ultralytics YOLO for object detection
- Pillow, NumPy, Matplotlib
- FFmpeg is used by helper code for HLS download and MP4/H.264 handling when
  available on the host

## Setup

From the backend directory:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env` from the example:

```bash
cp .env.example .env
```

Then fill in the TwelveLabs values you need:

```bash
TWELVELABS_API_KEY=your_twelvelabs_api_key_here
TWELVELABS_INDEX_ID=your_index_id_here
TWELVELABS_ENTITY_COLLECTION_ID=
SELF_APP_PING_URL=
SELF_APP_PING_INTERVAL_MINUTES=9
```

Run the API:

```bash
source .venv/bin/activate
python app.py
```

The app listens on `0.0.0.0:5001` in debug mode when run directly.

For a production-style process:

```bash
gunicorn "app:app" --bind 0.0.0.0:5001
```

## Configuration

Common environment variables:

| Variable | Required | Purpose |
| --- | --- | --- |
| `TWELVELABS_API_KEY` | Required for TwelveLabs features | Enables indexing, search, analysis, and entity operations. |
| `TWELVELABS_INDEX_ID` | Optional but recommended | Target TwelveLabs index. `config.py` has a hard-coded fallback for local development. |
| `TWELVELABS_ENTITY_COLLECTION_ID` | Optional | Existing entity collection. If empty, the backend resolves or creates `video-redaction-faces` when entity APIs are available. |
| `SELF_APP_PING_URL` | Optional | URL to ping periodically to keep a hosted backend warm. |
| `SELF_APP_PING_INTERVAL_MINUTES` | Optional | Self-ping interval. Defaults to `9`. |
| `SELF_APP_PING_TIMEOUT_SEC` | Optional | Self-ping timeout. Defaults to `15.0`. |
| `YOLO_OBJECT_MODEL` | Optional | Override the YOLO model path or name used for object detection. |
| `INSIGHTFACE_PROVIDERS` | Optional | Comma-separated ONNX Runtime providers for InsightFace. |

Advanced tracker tuning lives in `config.py`, including `TRACKER_MAX_DIM`,
`TRACKER_SMOOTHING_ALPHA`, `TRACKER_SIZE_SMOOTHING_ALPHA`,
`TRACKER_VELOCITY_SMOOTHING_ALPHA`, `TRACKER_PREDICTION_MAX_FRAMES`, and manual
face search expansion settings.

## Runtime Files

| Path | Purpose |
| --- | --- |
| `output/` | Uploaded source videos, temporary files, and redacted MP4 exports. |
| `snaps/<job_id>/job_manifest.json` | Persisted job status and metadata. |
| `snaps/<job_id>/detection_metadata.json` | Persisted face/object detection metadata. |
| `snaps/<job_id>/faces/` | Face snapshots for UI review and entity upload. |
| `snaps/<job_id>/objects/` | Object snapshots for UI review. |
| `snaps/<job_id>/face_lock_tracks/` | Precomputed per-person face-lock lanes. |
| `logs/pipeline.log` | Backend pipeline and route logs. |
| `models/deploy.prototxt` and `models/res10_300x300_ssd_iter_140000.caffemodel` | OpenCV DNN face detector files. |
| `yolov8n.pt` | Default local YOLO object detector weight. |

Generated `output/`, `snaps/`, `.cache/`, and log files are runtime artifacts.

## Health And Playback

### `GET /`

Health check.

```json
{
  "status": "ok",
  "service": "video-redaction-api"
}
```

### `GET /api/hls-proxy/<host>/<path>`

Proxies TwelveLabs/CloudFront HLS content so browsers can play indexed videos
without CDN CORS issues.

Allowed hosts must end with:

- `.cloudfront.net`
- `.twelvelabs.io`

Responses:

- `200` with rewritten `.m3u8` text or media segment bytes
- `403` for disallowed hosts
- `502` if the upstream fetch fails

## Local Processing Jobs

Use these endpoints when the backend needs local detections, snapshots, live
preview, or redacted export.

### `POST /api/index`

Upload a video and start the local processing pipeline. Unless skipped, this
also submits the video to TwelveLabs.

Input: `multipart/form-data`

| Field | Required | Notes |
| --- | --- | --- |
| `video` | Yes | Source video file. |
| `detect_interval_sec` | No | Keyframe interval when analysis timestamps are not available. Defaults to `1.0`. |
| `skip_indexing` | No | `true`, `1`, or `yes` skips TwelveLabs upload. |
| `video_id` | No | Existing TwelveLabs video id to bind when `skip_indexing` is true. |
| `from_job_id` | No | Reuses a previous job's TwelveLabs video id when possible. |

Response: `202 Accepted`

```json
{
  "job_id": "09d32fbb-b8a",
  "status": "processing",
  "message": "Video index started. Poll /api/index/<job_id> for status."
}
```

### `GET /api/index/<job_id>`

Poll a local processing job.

```json
{
  "job_id": "09d32fbb-b8a",
  "status": "ready",
  "twelvelabs_status": "done",
  "local_status": "done",
  "video_filename": "clip.mp4",
  "created_at": "2026-04-21T12:00:00+00:00",
  "error": null,
  "video_metadata": {
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "total_frames": 3600,
    "duration_sec": 120.0
  }
}
```

Returns `404` when the job is unknown.

### `GET /api/jobs`

Lists in-memory jobs for the current server process.

```json
{
  "jobs": [
    {
      "job_id": "09d32fbb-b8a",
      "status": "ready",
      "created_at": "2026-04-21T12:00:00+00:00",
      "video_filename": "clip.mp4"
    }
  ]
}
```

### `GET /api/jobs/by-video/<video_id>`

Find or create the best local job for a TwelveLabs video.

Query parameters:

| Parameter | Purpose |
| --- | --- |
| `ensure=true` | Create a local job if an exact/recovered job is not available. |
| `exact=true` | Only accept explicit `twelvelabs_video_id` matches. |
| `force=true` | Create a fresh job even when an existing failed job is found. |
| `recover=true` | Recover an unbound job only when the filename matches exactly. |

Response: `200 OK`, or `202 Accepted` when a new job was created.

```json
{
  "job_id": "09d32fbb-b8a",
  "status": "ready",
  "local_status": "done",
  "created": false,
  "forced": false
}
```

Returns `404` when no job can be found or created.

### `POST /api/jobs/<job_id>/push-entities`

Uploads the job's detected face snapshots to TwelveLabs as entities. The
pipeline does not do this automatically.

```json
{
  "job_id": "09d32fbb-b8a",
  "pushed": true,
  "entities_count": 2,
  "entities": [
    { "entity_id": "entity_123", "name": "person_0" }
  ]
}
```

Returns `503` when the TwelveLabs entity API is unavailable.

## TwelveLabs Indexing

These endpoints manage TwelveLabs indexing directly. They are separate from the
local `/api/index` pipeline.

### `POST /api/indexing`

Submit a video to TwelveLabs. Supports three input modes:

- `multipart/form-data` with `video=<file>`
- JSON with `video_url`
- JSON with `video_path` pointing to a local file

Optional query:

- `wait=true` waits synchronously for the TwelveLabs task to complete.

Async response: `202 Accepted`

```json
{
  "task_id": "task_123",
  "video_id": "video_123",
  "index_id": "index_123",
  "status": "submitted",
  "source": "upload",
  "message": "Indexing started. Poll /api/indexing/tasks/<task_id> for status."
}
```

### `POST /api/indexing/local`

Submit a local file path to TwelveLabs.

```json
{
  "video_path": "/absolute/path/to/video.mp4"
}
```

Response: `202 Accepted`, with the same task shape as `/api/indexing`.

### `GET /api/indexing/tasks`

Lists TwelveLabs tasks.

Query parameters:

- `index_id`
- `status`, comma-separated
- `page`, default `1`
- `page_limit`, default `10`

```json
{
  "tasks": [
    {
      "task_id": "task_123",
      "video_id": "video_123",
      "status": "ready",
      "index_id": "index_123",
      "system_metadata": {},
      "created_at": "2026-04-21T12:00:00Z",
      "updated_at": "2026-04-21T12:05:00Z"
    }
  ],
  "index_id": "index_123"
}
```

### `GET /api/indexing/tasks/<task_id>`

Retrieves a TwelveLabs task and adds local tracking status when the task was
submitted through this backend.

### `GET /api/indexing/videos`

Lists indexed videos from TwelveLabs.

Query parameters:

- `index_id`
- `page`, default `1`
- `page_limit`, default `10`

```json
{
  "videos": [
    {
      "video_id": "video_123",
      "system_metadata": {
        "filename": "clip.mp4",
        "duration": 120,
        "fps": 30,
        "width": 1920,
        "height": 1080
      },
      "hls_url": "https://...",
      "thumbnail_url": "https://...",
      "created_at": "2026-04-21T12:00:00Z",
      "updated_at": "2026-04-21T12:03:00Z",
      "indexed_at": "2026-04-21T12:05:00Z"
    }
  ],
  "index_id": "index_123"
}
```

### `GET /api/indexing/videos/<video_id>`

Retrieves one indexed video. Optional query: `index_id`.

### `DELETE /api/indexing/videos/<video_id>`

Deletes a video from the TwelveLabs index. Optional query: `index_id`.

```json
{
  "deleted": "video_123",
  "index_id": "index_123"
}
```

### `GET /api/indexing/info`

Retrieves index metadata. Optional query: `index_id`.

## Video Aliases

The frontend primarily uses these shorter aliases for indexed video metadata.

### `GET /api/videos`

Alias-style list of indexed videos. Query: `index_id`, `page`, `page_limit`.
The route also overlays a local filename when a matching local job exists.

### `GET /api/videos/<video_id>`

Retrieves one indexed video and adds `overview` from TwelveLabs `user_metadata`
when present.

### `POST /api/videos/<video_id>/overview`

Stores frontend-generated overview metadata on the TwelveLabs video.

```json
{
  "about": "One-line summary",
  "topics": ["topic-a", "topic-b"],
  "categories": ["Interview", "Demo"]
}
```

Response:

```json
{ "ok": true }
```

## Detection And Analysis

### `GET /api/faces/<job_id>`

Returns clustered face identities for a local job.

Responses:

- `200` when ready
- `202` while processing
- `409` if the job failed
- `404` if the job does not exist

```json
{
  "status": "ready",
  "unique_faces": [
    {
      "person_id": "person_0",
      "stable_person_id": "person_0",
      "name": "Person 1",
      "snap_base64": "iVBORw0KGgo...",
      "description": "Person description",
      "time_ranges": [{ "start": 12.4, "end": 18.8 }],
      "appearance_count": 5,
      "appearances": [
        { "timestamp": 12.4, "bbox": [10, 20, 110, 160] }
      ],
      "encoding": [],
      "bbox": [10, 20, 110, 160],
      "entity_id": "entity_123",
      "entity_asset_ids": [],
      "tags": [],
      "should_anonymize": false,
      "is_official": false,
      "priority_rank": 1
    }
  ],
  "entities": [],
  "total_face_detections": 5,
  "twelvelabs_people": [],
  "video_metadata": {}
}
```

### `GET /api/objects/<job_id>`

Returns clustered object classes for a local job.

```json
{
  "status": "ready",
  "unique_objects": [
    {
      "object_id": "object_0",
      "identification": "cell phone",
      "snap_base64": "iVBORw0KGgo...",
      "appearance_count": 4,
      "appearances": [
        { "timestamp": 2.1, "bbox": [10, 20, 110, 160] }
      ],
      "bbox": [10, 20, 110, 160]
    }
  ],
  "total_object_detections": 4,
  "twelvelabs_objects": [],
  "video_metadata": {}
}
```

### `GET /api/scene-summary/<job_id>`

Returns the TwelveLabs scene summary attached to a local job.

```json
{
  "status": "ready",
  "scene_summary": {},
  "video_metadata": {}
}
```

### `POST /api/analyze-custom`

Runs a custom TwelveLabs analysis prompt for an indexed video.

```json
{
  "video_id": "video_123",
  "prompt": "Summarize this video"
}
```

Response:

```json
{
  "data": "Model response text",
  "id": "analysis_123"
}
```

### `POST /api/detect-faces`

Detects faces in an uploaded reference image and returns cropped previews.

Input: `multipart/form-data`

```text
image=<binary image file>
```

Response:

```json
{
  "faces": [
    {
      "index": 0,
      "confidence": 0.98,
      "bbox": { "x": 120, "y": 80, "w": 160, "h": 160 },
      "image_base64": "iVBORw0KGgo...",
      "source": "insightface"
    }
  ]
}
```

### `POST /api/live-redaction/detect`

Returns normalized face/object boxes for the current playback time. Accepts JSON
or form fields.

```json
{
  "job_id": "09d32fbb-b8a",
  "time_sec": 12.4,
  "reset_tracking": false,
  "include_faces": true,
  "include_objects": true,
  "person_ids": ["person_0"],
  "object_classes": ["cell phone"],
  "forensic_only": true,
  "object_confidence": 0.25,
  "face_confidence": 0.28
}
```

Response:

```json
{
  "status": "ready",
  "job_id": "09d32fbb-b8a",
  "time_sec": 12.4,
  "detections": [
    {
      "id": "face-0",
      "trackId": "09d32fbb-b8a:0",
      "kind": "face",
      "label": "Person 1",
      "personId": "person_0",
      "confidence": 0.92,
      "x": 0.12,
      "y": 0.22,
      "width": 0.18,
      "height": 0.24
    }
  ],
  "object_detection_error": null,
  "frame": {
    "width": 1920,
    "height": 1080
  }
}
```

## Search

### `POST /api/search`

Searches indexed videos using text, uploaded image files, an image URL, or a
combination. JSON and `multipart/form-data` are supported.

JSON input:

```json
{
  "query": "person with blue jacket",
  "index_id": "index_123",
  "image_url": "https://example.com/reference.jpg",
  "operator": "and"
}
```

Multipart input:

```text
query=person with blue jacket
index_id=index_123
operator=and
image=<binary image file>      # repeatable
```

`operator` can be `and` or `or`. Invalid values are ignored.

Response:

```json
{
  "query": "person with blue jacket",
  "index_id": "index_123",
  "group_by": "video",
  "operator": "and",
  "results": [
    {
      "video_id": "video_123",
      "score": 0.93,
      "clips": [
        {
          "start": 12.4,
          "end": 18.8,
          "score": 0.93,
          "rank": 1,
          "thumbnail_url": "https://..."
        }
      ]
    }
  ]
}
```

### `POST /api/search/person-segments`

Finds time ranges matching a person description.

```json
{
  "video_id": "video_123",
  "description": "woman in a blue jacket"
}
```

Response:

```json
{
  "video_id": "video_123",
  "description": "woman in a blue jacket",
  "time_ranges": [
    { "start": 12.4, "end": 18.8 }
  ]
}
```

## Entities

Entity endpoints depend on TwelveLabs entity API availability. When unavailable,
list routes return an empty unavailable payload and mutating routes return `503`.

### `GET /api/entities`

```json
{
  "entity_collection_id": "collection_123",
  "entities": [
    {
      "id": "entity_123",
      "entity_id": "entity_123",
      "name": "John Doe",
      "status": "ready",
      "asset_ids": ["asset_123"],
      "metadata": {
        "name": "John Doe",
        "face_snap_base64": "iVBORw0KGgo..."
      }
    }
  ]
}
```

Unavailable response:

```json
{
  "entity_collection_id": null,
  "entities": [],
  "unavailable": true,
  "message": "TwelveLabs entity API is unavailable..."
}
```

### `GET /api/entities/<entity_id>`

Retrieves one entity.

### `POST /api/entities`

Creates an entity from existing asset IDs.

```json
{
  "name": "John Doe",
  "asset_ids": ["asset_123"],
  "description": "Known subject",
  "metadata": {
    "source": "operator"
  }
}
```

Response: `201 Created`

### `DELETE /api/entities/<entity_id>`

Deletes one entity.

```json
{ "deleted": "entity_123" }
```

### `POST /api/entities/upload-face`

Uploads a face image asset and creates an entity.

Input: `multipart/form-data`

```text
image=<binary image file>
name=John Doe
preview_base64=iVBORw0KGgo...   # optional
description=Known subject      # optional
```

Response: `201 Created`

```json
{
  "asset_id": "asset_123",
  "entity": {
    "id": "entity_123",
    "entity_id": "entity_123",
    "name": "John Doe",
    "status": "ready",
    "asset_ids": ["asset_123"],
    "metadata": {
      "name": "John Doe",
      "face_snap_base64": "iVBORw0KGgo..."
    }
  }
}
```

### `POST /api/entities/<entity_id>/add-asset`

Adds another face asset to an entity. Supports:

- `multipart/form-data` with `image=<file>`
- JSON with `image_url`

```json
{
  "entity_id": "entity_123",
  "new_asset_id": "asset_456",
  "total_assets": 2,
  "entity": {}
}
```

### `POST /api/entities/<entity_id>/search`

Searches videos with an entity token, optionally with extra text.

```json
{
  "query": "wearing a hat",
  "index_id": "index_123"
}
```

Response:

```json
{
  "entity_id": "entity_123",
  "query_suffix": "wearing a hat",
  "index_id": "index_123",
  "group_by": "video",
  "results": []
}
```

### `POST /api/entities/<entity_id>/time-ranges`

Returns search-derived time ranges for an entity.

```json
{
  "video_id": "video_123",
  "index_id": "index_123"
}
```

Response:

```json
{
  "entity_id": "entity_123",
  "video_id": "video_123",
  "time_ranges": [
    { "start": 12.4, "end": 18.8 }
  ]
}
```

### `GET /api/entity-collections`

Lists available TwelveLabs entity collections.

```json
{
  "collections": []
}
```

## Redaction And Export

### Redaction Request Body

Both `/api/redact` and `/api/redact/jobs` use the same request body. JSON and
form fields are accepted.

```json
{
  "job_id": "09d32fbb-b8a",
  "person_ids": ["person_0"],
  "face_encodings": [],
  "object_classes": ["cell phone"],
  "entity_ids": ["entity_123"],
  "custom_regions": [
    {
      "id": "region_1",
      "x": 0.1,
      "y": 0.2,
      "width": 0.15,
      "height": 0.2,
      "shape": "rectangle",
      "effect": "blur",
      "anchor_sec": 4.5,
      "reason": "screen",
      "tracking_mode": "region"
    }
  ],
  "blur_strength": 60,
  "redaction_style": "blur",
  "detect_every_n": 3,
  "detect_every_seconds": null,
  "use_temporal_optimization": true,
  "output_height": 720
}
```

Notes:

- At least one of `person_ids`, `face_encodings`, `object_classes`,
  `entity_ids`, or `custom_regions` is required.
- `redaction_style` supports `blur` and `black`; invalid values fall back to
  `blur`.
- Export quality accepts `480p`, `720p`, `1080p`, or numeric heights
  `480`, `720`, `1080`.
- `export_height` and `export_quality` are accepted aliases for
  `output_height`.
- If selected face targets or face-like custom regions are present, the backend
  forces per-frame detection for better face coverage.

### `POST /api/redact`

Runs redaction synchronously and returns when rendering is complete.

```json
{
  "output_path": "/absolute/path/to/redacted.mp4",
  "download_url": "/api/download/redacted_09d32fbb-b8a_720p.mp4",
  "download_filename": "redacted_09d32fbb-b8a_720p.mp4",
  "mime_type": "video/mp4",
  "output_size_bytes": 1234567,
  "h264_encoded": true,
  "download_ready": true,
  "export_quality": "720p",
  "width": 1280,
  "height": 720,
  "total_frames": 3600,
  "fps": 30,
  "detection_frames_processed": 1200,
  "detection_frames_skipped": 2400,
  "entity_ids_used": [],
  "temporal_ranges_from_entity_search": 0,
  "person_ids_used": ["person_0"]
}
```

### `POST /api/redact/jobs`

Starts an asynchronous redaction render.

Response: `202 Accepted`

```json
{
  "redaction_job_id": "redact_123",
  "status": "queued",
  "stage": "queued",
  "progress": 0.0,
  "percent": 0,
  "message": "Queued for rendering"
}
```

### `GET /api/redact/jobs/<redaction_job_id>`

Polls an asynchronous redaction job.

```json
{
  "redaction_job_id": "redact_123",
  "source_job_id": "09d32fbb-b8a",
  "status": "completed",
  "stage": "completed",
  "progress": 1.0,
  "percent": 100,
  "frames_processed": 3600,
  "total_frames": 3600,
  "message": "Redaction complete",
  "error": null,
  "result": {
    "download_url": "/api/download/redacted_09d32fbb-b8a_720p.mp4"
  }
}
```

### `POST /api/redact/preview-track`

Builds preview samples for custom region tracking.

```json
{
  "job_id": "09d32fbb-b8a",
  "preview_fps": 8,
  "custom_regions": [
    {
      "id": "region_1",
      "x": 0.1,
      "y": 0.2,
      "width": 0.15,
      "height": 0.2,
      "shape": "rectangle",
      "effect": "blur",
      "anchor_sec": 4.5,
      "reason": "screen"
    }
  ]
}
```

Response:

```json
{
  "custom_tracks": [
    {
      "id": "region_1",
      "samples": [
        { "t": 4.5, "x": 0.1, "y": 0.2, "width": 0.15, "height": 0.2 }
      ]
    }
  ],
  "fps": 30,
  "width": 1920,
  "height": 1080,
  "total_frames": 3600
}
```

### `GET /api/download/<filename>`

Downloads a redacted MP4 from `output/`.

Important behavior:

- Only safe redacted MP4 filenames are accepted.
- A normal full-file `GET` streams the file and removes it after the response
  finishes.
- Range requests are served with `send_file` for media compatibility.

Responses:

- `200` with `video/mp4`
- `400` for unsupported filenames
- `404` when the file does not exist
- `410` when the MP4 is not valid or not ready

### `GET /api/thumbnails/<video_id>.jpg`

Serves a generated frontend thumbnail from
`frontend/public/generated-thumbnails/<video_id>.jpg`.

Returns `404` when the thumbnail is missing.

## Face-Lock Tracks

Face-lock tracks are precomputed per-person bbox lanes used by preview/export
logic for steadier selected-face redaction.

### `POST /api/face-lock-track/build`

Queues a background face-lock lane build.

```json
{
  "job_id": "09d32fbb-b8a",
  "person_id": "person_0",
  "force_rebuild": false
}
```

Response: `202 Accepted`, or `200 OK` when a cached lane is already available.

```json
{
  "status": "queued",
  "progress": 0.0,
  "percent": 0,
  "job_id": "09d32fbb-b8a",
  "person_id": "person_0",
  "queued": true
}
```

### `GET /api/face-lock-track/<job_id>/<person_id>`

Retrieves build status and, by default, the full lane.

Query parameters:

- `include_lane=false` returns only lane metadata instead of full samples.

```json
{
  "job_id": "09d32fbb-b8a",
  "person_id": "person_0",
  "status": "ready",
  "progress": 1.0,
  "percent": 100,
  "message": null,
  "cached": true,
  "lane": {}
}
```

Returns `500` when the build status is `failed`.

