import os
from typing import Callable, TypeVar

from dotenv import load_dotenv

T = TypeVar("T")


def env_cast(name: str, default: T, parser: Callable[[str], T]) -> T:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return parser(raw)
    except ValueError:
        return default

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(CONFIG_DIR, ".env")
if os.path.isfile(ENV_PATH):
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

TWELVELABS_API_KEY = os.environ.get("TWELVELABS_API_KEY", "").strip()
TWELVELABS_INDEX_ID = (os.environ.get("TWELVELABS_INDEX_ID") or "699d11012e158988856161cc").strip()
TWELVELABS_ENTITY_COLLECTION_ID = (os.environ.get("TWELVELABS_ENTITY_COLLECTION_ID") or "").strip()
TWELVELABS_ENTITY_COLLECTION_NAME = "video-redaction-faces"
SELF_APP_PING_URL = (os.environ.get("SELF_APP_PING_URL") or "").strip()
SELF_APP_PING_INTERVAL_MINUTES = env_cast("SELF_APP_PING_INTERVAL_MINUTES", 9, int)
SELF_APP_PING_TIMEOUT_SEC = env_cast("SELF_APP_PING_TIMEOUT_SEC", 15.0, float)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPS_DIR = os.path.join(BASE_DIR, "snaps")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

FACE_COSINE_SIM_THRESHOLD = 0.42
OBJECT_CONF_THRESHOLD = 0.25
DEFAULT_BLUR_STRENGTH = 220
DEFAULT_DETECT_EVERY_N = 1
DEFAULT_DETECT_INTERVAL_SEC = 1.0
TRACKER_MAX_DIM = env_cast("TRACKER_MAX_DIM", 960, int)
MIN_TRACKER_ROI_PIXELS = 20
TRACKER_REINIT_BBOX_EXPAND_FACTOR = env_cast("TRACKER_REINIT_BBOX_EXPAND_FACTOR", 1.15, float)
TRACKER_SMOOTHING_ALPHA = env_cast("TRACKER_SMOOTHING_ALPHA", 0.55, float)
TRACKER_SIZE_SMOOTHING_ALPHA = env_cast("TRACKER_SIZE_SMOOTHING_ALPHA", 0.4, float)
TRACKER_VELOCITY_SMOOTHING_ALPHA = env_cast("TRACKER_VELOCITY_SMOOTHING_ALPHA", 0.45, float)
TRACKER_PREDICTION_MAX_FRAMES = env_cast("TRACKER_PREDICTION_MAX_FRAMES", 30, int)
TRACK_ASSOCIATION_IOU = env_cast("TRACK_ASSOCIATION_IOU", 0.28, float)
TRACK_ASSOCIATION_CENTER_RATIO = env_cast("TRACK_ASSOCIATION_CENTER_RATIO", 0.5, float)
TRACK_LOST_GRACE_FRAMES = env_cast("TRACK_LOST_GRACE_FRAMES", 45, int)
MANUAL_FACE_SEARCH_EXPAND_FACTOR = env_cast("MANUAL_FACE_SEARCH_EXPAND_FACTOR", 2.0, float)
MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR = env_cast("MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR", 3.0, float)
MANUAL_FACE_DETECTION_CONFIDENCE = env_cast("MANUAL_FACE_DETECTION_CONFIDENCE", 0.16, float)
KEYFRAME_INTERVAL_SEC = 1.0
