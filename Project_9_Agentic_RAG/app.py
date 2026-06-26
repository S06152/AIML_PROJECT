"""
Agentic RAG Knowledge Assistant - Single Entry Point.

Works in both environments:
    - Streamlit Cloud: streamlit run app.py
      (Auto-starts FastAPI backend in a background thread)
    - Local Development: python app.py
      (Starts standalone FastAPI backend)
"""

import sys
import time
import threading
import warnings

warnings.filterwarnings("ignore")

from src.utils.logger import logging
from src.utils.exception import CustomException


# ─── Detect Runtime Environment ────────────────────────────────────────────────

def _is_running_under_streamlit() -> bool:
    """Check if the script is being executed by Streamlit runtime."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# ─── FastAPI Backend Helpers ───────────────────────────────────────────────────

FASTAPI_PORT = 8000
FASTAPI_URL = f"http://localhost:{FASTAPI_PORT}"


def _start_fastapi_server():
    """Start FastAPI/uvicorn server in a background thread."""
    import uvicorn
    from src.api.fastapi_app import app as fastapi_app

    logging.info("Starting FastAPI backend on port %d (background thread)...", FASTAPI_PORT)
    uvicorn.run(fastapi_app, host="0.0.0.0", port=FASTAPI_PORT, log_level="warning")


def _is_backend_running() -> bool:
    """Check if FastAPI backend is already running."""
    try:
        import requests
        resp = requests.get(f"{FASTAPI_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _ensure_backend_running():
    """
    Start FastAPI backend in a daemon thread if not already running.
    Waits until the server is ready to accept requests.
    """
    if _is_backend_running():
        logging.info("FastAPI backend is already running.")
        return

    server_thread = threading.Thread(target=_start_fastapi_server, daemon=True)
    server_thread.start()

    for i in range(30):
        time.sleep(0.5)
        if _is_backend_running():
            logging.info("FastAPI backend is ready (took ~%.1f seconds).", (i + 1) * 0.5)
            return

    logging.warning("FastAPI backend did not respond in time, but may still be starting.")


# ─── Main Execution ───────────────────────────────────────────────────────────

if _is_running_under_streamlit():
    # ── Streamlit Cloud / streamlit run app.py ──
    try:
        logging.info("APPLICATION START - Streamlit Cloud Mode")

        # Step 1: Auto-start FastAPI backend in background thread
        _ensure_backend_running()

        # Step 2: Launch Streamlit frontend
        from src.ui.streamlit_frontend import StreamlitFrontend

        logging.info("Launching Streamlit Frontend...")
        frontend = StreamlitFrontend()
        frontend.run()

        logging.info("APPLICATION END")

    except Exception as e:
        logging.exception("CRITICAL ERROR: Failed to launch application.")
        raise CustomException(e, sys)

else:
    # ── Local Development: python app.py ──
    if __name__ == "__main__":
        try:
            import uvicorn

            logging.info("APPLICATION START - FastAPI Backend Mode (Local)")
            logging.info("Starting FastAPI backend server...")

            uvicorn.run(
                "src.api.fastapi_app:app",
                host="0.0.0.0",
                port=FASTAPI_PORT,
                reload=True,
            )

            logging.info("APPLICATION END")

        except Exception as e:
            logging.exception("CRITICAL ERROR: Failed to start FastAPI backend.")
            raise CustomException(e, sys)