import sys
import time
import threading
import warnings
import requests
import uvicorn
from src.ui.fastapi import app as fastapi_app
from src.main import AGENTICRAG
from src.utils.exception import CustomException
from src.utils.logger import logging
warnings.filterwarnings("ignore")

# FastAPI Configuration
FASTAPI_PORT = 8000
FASTAPI_URL = f"http://localhost:{FASTAPI_PORT}"

def _start_fastapi_server():
    """
    Start FastAPI server in a background thread.
    """
    try:
        logging.info("Starting FastAPI backend on port %d", FASTAPI_PORT)

        uvicorn.run(fastapi_app, host = "0.0.0.0", port = FASTAPI_PORT, log_level = "warning")

    except Exception as e:
        logging.exception("Error while starting FastAPI backend.")
        raise CustomException(e, sys)
    
def _is_backend_running() -> bool:
    """
    Check whether the FastAPI backend is running.
    """
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout = 2)

        return response.status_code == 200

    except requests.RequestException:
        # Backend is not running yet.
        return False

    except Exception as e:
        logging.exception("Error while checking backend status.")
        raise CustomException(e, sys)

def _ensure_backend_running():
    """
    Start FastAPI backend if it is not already running.
    """
    try:
        logging.info("Checking FastAPI backend status...")

        # Backend is already running
        if _is_backend_running():
            logging.info("FastAPI backend is already running.")
            return

        logging.info("FastAPI backend not found. Starting backend...")

        # Start backend in background
        server_thread = threading.Thread(target = _start_fastapi_server, daemon = True)
        server_thread.start()

        # Wait until backend is ready
        for i in range(30):
            time.sleep(0.5)

            if _is_backend_running():
                logging.info("FastAPI backend is ready (took ~%.1f seconds).", (i + 1) * 0.5)
                return

        logging.warning("FastAPI backend did not respond within the expected time.")

    except Exception as e:
        logging.exception("Error while ensuring backend is running.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        logging.info("Application started.")

        # Step 1: Start FastAPI backend
        _ensure_backend_running()

        # Step 2: Launch Streamlit frontend
        logging.info("Launching Streamlit frontend")

        app = AGENTICRAG()
        app.run()

        logging.info("Application completed successfully")

    except Exception as e:
        logging.exception("Application failed to start")
        raise CustomException(e, sys)
















