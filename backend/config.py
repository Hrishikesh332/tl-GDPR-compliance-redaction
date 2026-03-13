import os

from dotenv import load_dotenv

# Load .env from the same directory as this config file (backend/.env)
# so it works regardless of current working directory.
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_CONFIG_DIR, ".env")
if os.path.isfile(_ENV_PATH):
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()  # fallback: search cwd and parents

TWELVELABS_API_KEY = os.environ.get("TWELVELABS_API_KEY", "").strip()
TWELVELABS_INDEX_ID = (os.environ.get("TWELVELABS_INDEX_ID") or "699d11012e158988856161cc").strip()
TWELVELABS_ENTITY_COLLECTION_ID = (os.environ.get("TWELVELABS_ENTITY_COLLECTION_ID") or "").strip()
TWELVELABS_ENTITY_COLLECTION_NAME = "video-redaction-faces"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPS_DIR = os.path.join(BASE_DIR, "snaps")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

FACE_COSINE_SIM_THRESHOLD = 0.55
OBJECT_CONF_THRESHOLD = 0.25
DEFAULT_BLUR_STRENGTH = 51
DEFAULT_DETECT_EVERY_N = 10
DEFAULT_DETECT_INTERVAL_SEC = 1.0
TRACKER_MAX_DIM = 480
# Minimum tracker ROI side in scaled image (trackers often fail on very small ROIs)
MIN_TRACKER_ROI_PIXELS = 20
# Re-init bbox expansion (e.g. 1.15 = 15% larger) so blur does not clip after camera zoom
TRACKER_REINIT_BBOX_EXPAND_FACTOR = float(os.environ.get("TRACKER_REINIT_BBOX_EXPAND_FACTOR", "1.15"))
# EMA smoothing of tracker bbox (0 = off, 0.2–0.4 = smoother). New bbox = alpha*raw + (1-alpha)*prev
TRACKER_SMOOTHING_ALPHA = float(os.environ.get("TRACKER_SMOOTHING_ALPHA", "0.0"))
MANUAL_FACE_SEARCH_EXPAND_FACTOR = float(os.environ.get("MANUAL_FACE_SEARCH_EXPAND_FACTOR", "1.65"))
MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR = float(os.environ.get("MANUAL_FACE_LOST_SEARCH_EXPAND_FACTOR", "2.25"))
MANUAL_FACE_DETECTION_CONFIDENCE = float(os.environ.get("MANUAL_FACE_DETECTION_CONFIDENCE", "0.22"))
KEYFRAME_INTERVAL_SEC = 1.0
