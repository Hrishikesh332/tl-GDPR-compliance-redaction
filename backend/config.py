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

FACE_COSINE_SIM_THRESHOLD = 0.55
OBJECT_CONF_THRESHOLD = 0.25
DEFAULT_BLUR_STRENGTH = 51
DEFAULT_DETECT_EVERY_N = 10
DEFAULT_DETECT_INTERVAL_SEC = 1.0
TRACKER_MAX_DIM = 480
MIN_TRACKER_ROI_PIXELS = 20
TRACKER_REINIT_BBOX_EXPAND_FACTOR = _env_cast("TRACKER_REINIT_BBOX_EXPAND_FACTOR", 1.15, float)
TRACKER_SMOOTHING_ALPHA = _env_cast("TRACKER_SMOOTHING_ALPHA", 0.0, float)
MANUAL_FACE_SEARCH_EXPAND_FACTOR = _env_cast("MANUAL_FACE_SEARCH_EXPAND_FACTOR", 1.65, float)
MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR = _env_cast("MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR", 2.25, float)
MANUAL_FACE_DETECTION_CONFIDENCE = _env_cast("MANUAL_FACE_DETECTION_CONFIDENCE", 0.22, float)
KEYFRAME_INTERVAL_SEC = 1.0
