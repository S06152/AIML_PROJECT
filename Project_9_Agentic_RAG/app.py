# Standard Library Imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.main import MultiModalRAG
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    try:
        logging.info("APPLICATION START")
        logging.info("Initializing Multi-Modal RAG Pipeline (Streamlit App)")

        # Initialize Application
        logging.info("Creating Streamlit application instance...")
        app = MultiModalRAG()

        logging.info("Application instance created successfully | Class = %s", type(app).__name__)

        # Run Application
        logging.info("Launching Streamlit UI...")
        app.run()
        
        logging.info("Streamlit UI executed successfully.")
        logging.info("APPLICATION END")

    except Exception as e:
        logging.exception("CRITICAL ERROR: Failed to launch Multi-Modal RAG Pipeline application.")
        raise CustomException(e, sys)