"""
Streamlit Cloud Entry Point for Agentic RAG Knowledge Assistant.

Usage (Streamlit Cloud / Local):
    streamlit run streamlit_app.py

This single file:
    1. Starts the FastAPI backend automatically in a background thread.
    2. Launches the Streamlit frontend UI.

No separate terminal or manual server start is needed.
Just commit to GitHub, link to Streamlit Cloud, and it works.
"""

import sys
import time
import threading
import requests
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.ui.streamlit_frontend import StreamlitFrontend
import warnings

warnings.filterwarnings("ignore")

# ─── Auto-start FastAPI Backend in Background Thread ───────────────────────────

FASTAPI_PORT = 8000
FASTAPI_URL = f"http://localhost:{FASTAPI_PORT}"


def _start_fastapi_server():
    """Start FastAPI/uvicorn server in a background thread."""
    import uvicorn
    from src.api.fastapi_app import app as fastapi_app

    logging.info("Starting FastAPI backend server on port %d (background thread)...", FASTAPI_PORT)
    uvicorn.run(fastapi_app, host="0.0.0.0", port=FASTAPI_PORT, log_level="warning")


def _is_backend_running() -> bool:
    """Check if FastAPI backend is already running."""
    try:
        resp = requests.get(f"{FASTAPI_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _ensure_backend_running():
    """
    Ensure the FastAPI backend is running.

    If not already running, starts it in a daemon background thread
    and waits until it is ready to accept requests.
    """
    if _is_backend_running():
        logging.info("FastAPI backend is already running.")
        return

    # Start server in a daemon thread (auto-killed when main process exits)
    server_thread = threading.Thread(target=_start_fastapi_server, daemon=True)
    server_thread.start()

    # Wait for the server to become ready (max ~15 seconds)
    for i in range(30):
        time.sleep(0.5)
        if _is_backend_running():
            logging.info("FastAPI backend is ready (took ~%.1f seconds).", (i + 1) * 0.5)
            return

    logging.warning("FastAPI backend did not respond in time, but may still be starting.")


# ─── Launch Application ───────────────────────────────────────────────────────

try:
    logging.info("APPLICATION START - Streamlit Cloud Mode")

    # Step 1: Auto-start FastAPI backend
    _ensure_backend_running()

    # Step 2: Launch Streamlit frontend
    logging.info("Launching Streamlit Frontend...")
    frontend = StreamlitFrontend()
    frontend.run()

    logging.info("APPLICATION END")

except Exception as e:
    logging.exception(
        "CRITICAL ERROR: Failed to launch Agentic RAG application."
    )
    raise CustomException(e, sys)
