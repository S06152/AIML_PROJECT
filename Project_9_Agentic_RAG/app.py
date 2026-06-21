import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.main import AGENTICRAG
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    try:
        logging.info("APPLICATION START")
        logging.info("Initializing Agentic RAG Knowledge Assistant with External Tool Integration(Streamlit App)")

        # Initialize Application
        logging.info("Creating Streamlit application instance...")
        app = AGENTICRAG()

        logging.info("Application instance created successfully | Class = %s", type(app).__name__)

        # Run Application
        logging.info("Launching Streamlit UI...")
        app.run()
        
        logging.info("Streamlit UI executed successfully.")
        logging.info("APPLICATION END")

    except Exception as e:
        logging.exception("CRITICAL ERROR: Failed to launch Agentic RAG Knowledge Assistant with External Tool Integration application.")
        raise CustomException(e, sys)