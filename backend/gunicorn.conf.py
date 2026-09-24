"""One process owns the app's in-memory jobs; threads serve polling requests."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5001')}"
workers = 1
worker_class = "gthread"
threads = 8
timeout = 600
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
capture_output = True
