import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS

from config import BASE_DIR, SNAPS_DIR, OUTPUT_DIR, TWELVELABS_API_KEY, TWELVELABS_INDEX_ID
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


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
