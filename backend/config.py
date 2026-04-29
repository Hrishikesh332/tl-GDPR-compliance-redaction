import os
from typing import Callable, TypeVar

from dotenv import load_dotenv

T = TypeVar("T")


def _env_cast(name: str, default: T, parser: Callable[[str], T]) -> T:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return parser(raw)
    except ValueError:
        return default

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_CONFIG_DIR, ".env")
if os.path.isfile(_ENV_PATH):
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

TWELVELABS_API_KEY = os.environ.get("TWELVELABS_API_KEY", "").strip()
TWELVELABS_INDEX_ID = (os.environ.get("TWELVELABS_INDEX_ID") or "699d11012e158988856161cc").strip()
TWELVELABS_ENTITY_COLLECTION_ID = (os.environ.get("TWELVELABS_ENTITY_COLLECTION_ID") or "").strip()
TWELVELABS_ENTITY_COLLECTION_NAME = "video-redaction-faces"
SELF_APP_PING_URL = (os.environ.get("SELF_APP_PING_URL") or "").strip()
SELF_APP_PING_INTERVAL_MINUTES = _env_cast("SELF_APP_PING_INTERVAL_MINUTES", 9, int)
SELF_APP_PING_TIMEOUT_SEC = _env_cast("SELF_APP_PING_TIMEOUT_SEC", 15.0, float)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPS_DIR = os.path.join(BASE_DIR, "snaps")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

FACE_COSINE_SIM_THRESHOLD = 0.42
OBJECT_CONF_THRESHOLD = 0.25
DEFAULT_BLUR_STRENGTH = 220
# Frequent re-detection passes keep the blur anchored to the actual face
# even when the camera is panning, shaking, or zooming. The per-frame face
# search inside each track's search region also re-locks the blur when the
# face moves between passes.
DEFAULT_DETECT_EVERY_N = 1
DEFAULT_DETECT_INTERVAL_SEC = 1.0
# Higher tracking resolution preserves face-scale precision so the blur
# follows the head accurately when it grows or shrinks under zoom.
TRACKER_MAX_DIM = _env_cast("TRACKER_MAX_DIM", 960, int)
MIN_TRACKER_ROI_PIXELS = 20
TRACKER_REINIT_BBOX_EXPAND_FACTOR = _env_cast("TRACKER_REINIT_BBOX_EXPAND_FACTOR", 1.15, float)
# Position smoothing weight for the new measurement in EMA: higher = more
# responsive (less lag), lower = smoother. 0.55 balances responsiveness
# during pans against frame-to-frame jitter.
TRACKER_SMOOTHING_ALPHA = _env_cast("TRACKER_SMOOTHING_ALPHA", 0.55, float)
# Size smoothing weight: kept lower so blur grows/shrinks more steadily
# while the face zooms in or out (size flickers far less than position).
TRACKER_SIZE_SMOOTHING_ALPHA = _env_cast("TRACKER_SIZE_SMOOTHING_ALPHA", 0.4, float)
# Velocity smoothing for the constant-velocity Kalman-like predictor that
# keeps the blur locked on the face during brief detection drop-outs and
# fast camera shifts.
TRACKER_VELOCITY_SMOOTHING_ALPHA = _env_cast("TRACKER_VELOCITY_SMOOTHING_ALPHA", 0.45, float)
# Carry the blur on velocity prediction for up to ~0.4s at 30fps so brief
# detection blanks during fast camera shifts never reveal the face.
TRACKER_PREDICTION_MAX_FRAMES = _env_cast("TRACKER_PREDICTION_MAX_FRAMES", 30, int)
# IoU threshold used to associate fresh face detections to an existing
# track at a detection pass so the tracker, smoothing state, and identity
# survive across detection refreshes (no jitter every N frames).
TRACK_ASSOCIATION_IOU = _env_cast("TRACK_ASSOCIATION_IOU", 0.28, float)
TRACK_ASSOCIATION_CENTER_RATIO = _env_cast("TRACK_ASSOCIATION_CENTER_RATIO", 0.5, float)
# Keep recently-seen subjects alive for ~0.5s after a missed detection
# pass so the blur stays on faces during quick pans/zooms.
TRACK_LOST_GRACE_FRAMES = _env_cast("TRACK_LOST_GRACE_FRAMES", 45, int)
MANUAL_FACE_SEARCH_EXPAND_FACTOR = _env_cast("MANUAL_FACE_SEARCH_EXPAND_FACTOR", 2.0, float)
MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR = _env_cast("MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR", 3.0, float)
MANUAL_FACE_DETECTION_CONFIDENCE = _env_cast("MANUAL_FACE_DETECTION_CONFIDENCE", 0.16, float)
KEYFRAME_INTERVAL_SEC = 1.0
