# Standard Library Imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.ui.streamlit_app import StreamlitApp

if __name__ == "__main__":

    try:
        logging.info("Starting Streamlit RAG Application.")

        # Initialize Streamlit Application
        app = StreamlitApp()
        logging.info("StreamlitApp instance created successfully.")

        # Load and render the Streamlit UI
        app.load_streamlit_ui()
        logging.info("Streamlit UI loaded successfully.")

    except Exception as e:
        logging.exception("Fatal error occurred while launching the application.")
        raise CustomException(e, sys)