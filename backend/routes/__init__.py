from routes.ingest import index_bp
from routes.analysis import analysis_bp
from routes.search import search_bp
from routes.redaction import redaction_bp
from routes.download import download_bp
from routes.entity import entity_bp
from routes.indexing import indexing_bp
from routes.face_lock import face_lock_bp


def register_blueprints(app):
    app.register_blueprint(index_bp, url_prefix="/api")
    app.register_blueprint(analysis_bp, url_prefix="/api")
    app.register_blueprint(search_bp, url_prefix="/api")
    app.register_blueprint(redaction_bp, url_prefix="/api")
    app.register_blueprint(download_bp, url_prefix="/api")
    app.register_blueprint(entity_bp, url_prefix="/api")
    app.register_blueprint(indexing_bp, url_prefix="/api")
    app.register_blueprint(face_lock_bp, url_prefix="/api")
