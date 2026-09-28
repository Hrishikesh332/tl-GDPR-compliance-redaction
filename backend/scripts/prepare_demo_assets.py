"""Build-time entry point for the allowlisted, versioned demo bundle."""
from pathlib import Path
import sys

backend = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend))
from services.demo_data import prepare_demo_assets

prepare_demo_assets(backend)
