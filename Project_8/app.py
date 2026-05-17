import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.main import MultiModalRAGSystem

if __name__ == "__main__":
    try:
        logging.info("Starting Multi-Modal RAG System Application.")

        # Initialize Multi-Modal RAG System
        app = MultiModalRAGSystem()

        logging.info("Multi-Modal RAG System initialized successfully.")

        # Launch Streamlit UI
        app.run()

        logging.info("Streamlit UI for Multi-Modal RAG System launched successfully.")

    except Exception as e:
        logging.exception("Fatal error occurred while launching the Multi-Modal RAG System.")
        raise CustomException(e, sys)