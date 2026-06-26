"""
Agentic RAG Knowledge Assistant - FastAPI Backend Entry Point.

Usage:
    Standalone backend (for local dev):
        python app.py

    For Streamlit Cloud deployment, use streamlit_app.py instead:
        streamlit run streamlit_app.py
    (This auto-starts the FastAPI backend in a background thread)
"""

import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import warnings

warnings.filterwarnings("ignore")

# ─── FastAPI Application (for uvicorn) ─────────────────────────────────────────
from src.api.fastapi_app import app as fastapi_app


if __name__ == "__main__":
    """
    When run directly, starts the FastAPI backend server using uvicorn.
    """
    try:
        import uvicorn

        logging.info("APPLICATION START - FastAPI Backend Mode")
        logging.info("Starting Agentic RAG Knowledge Assistant FastAPI Backend...")

        uvicorn.run(
            "src.api.fastapi_app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
        )

        logging.info("APPLICATION END")

    except Exception as e:
        logging.exception(
            "CRITICAL ERROR: Failed to start FastAPI backend."
        )
        raise CustomException(e, sys)