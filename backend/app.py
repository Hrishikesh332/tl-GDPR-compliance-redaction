import logging
import os
import threading
from atexit import register as register_atexit

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify
from flask_cors import CORS

from config import (
    BASE_DIR,
    SNAPS_DIR,
    OUTPUT_DIR,
    SELF_APP_PING_INTERVAL_MINUTES,
    SELF_APP_PING_TIMEOUT_SEC,
    SELF_APP_PING_URL,
    TWELVELABS_API_KEY,
    TWELVELABS_INDEX_ID,
)
from routes import register_blueprints

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

os.makedirs(LOG_DIR, exist_ok=True)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
)
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger("video_redaction")
_self_ping_scheduler = None
_self_ping_scheduler_lock = threading.Lock()


def _shutdown_self_ping_scheduler():
    global _self_ping_scheduler
    scheduler = _self_ping_scheduler
    _self_ping_scheduler = None
    if scheduler is None:
        return
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        logger.debug("Self-ping scheduler shutdown skipped", exc_info=True)


def _start_self_ping_scheduler():
    global _self_ping_scheduler
    if not SELF_APP_PING_URL:
        return

    with _self_ping_scheduler_lock:
        if _self_ping_scheduler is not None:
            return

        scheduler = BackgroundScheduler(timezone="UTC", daemon=True)

        def _ping_self_app():
            try:
                response = requests.get(SELF_APP_PING_URL, timeout=SELF_APP_PING_TIMEOUT_SEC)
                logger.info(
                    "Self-ping to %s returned %s",
                    SELF_APP_PING_URL,
                    response.status_code,
                )
            except requests.RequestException as exc:
                logger.warning("Self-ping to %s failed: %s", SELF_APP_PING_URL, exc)

        scheduler.add_job(
            _ping_self_app,
            trigger="interval",
            minutes=max(1, SELF_APP_PING_INTERVAL_MINUTES),
            id="self-app-ping",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        scheduler.start()
        _self_ping_scheduler = scheduler
        logger.info(
            "Started self-ping scheduler for %s every %d minutes",
            SELF_APP_PING_URL,
            max(1, SELF_APP_PING_INTERVAL_MINUTES),
        )


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
    CORS(app)

    os.makedirs(SNAPS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if TWELVELABS_API_KEY:
        logger.info("Config: TWELVELABS_API_KEY loaded (length=%d)", len(TWELVELABS_API_KEY))
        logger.info("Config: TWELVELABS_INDEX_ID=%s", TWELVELABS_INDEX_ID)
    else:
        logger.warning("Config: TWELVELABS_API_KEY is empty — set it in backend/.env for TwelveLabs indexing")

    register_blueprints(app)
    _start_self_ping_scheduler()

    @app.route("/")
    def health():
        return jsonify({"status": "ok", "service": "video-redaction-api"})

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        logger.error("Internal error: %s", str(e))
        return jsonify({"error": "internal server error"}), 500

    return app


register_atexit(_shutdown_self_ping_scheduler)
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
