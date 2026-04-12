"""
app.py — Streamlit application entry point for AUTOSAR MAS.

Run with:
    streamlit run app.py

This file is intentionally minimal — all logic lives in src/main.py.
"""

from src.main import AutosarMASOrchestrator


def main() -> None:
    """Initialize and run the AUTOSAR MAS Streamlit application."""
    orchestrator = AutosarMASOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
