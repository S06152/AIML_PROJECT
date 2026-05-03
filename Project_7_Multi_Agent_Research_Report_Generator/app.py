# Standard Library Imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.main import Load_Multi_Agent_Research_Report_Generator
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    try:
        logging.info("APPLICATION START")
        logging.info("Initializing Multi-Agent Research & Report Generator (Streamlit App)")

        # Initialize Application
        logging.info("Creating Streamlit application instance...")
        app = Load_Multi_Agent_Research_Report_Generator()

        logging.info("Application instance created successfully | Class = %s", type(app).__name__)

        # Run Application
        logging.info("Launching Streamlit UI...")
        app.run()
        
        logging.info("Streamlit UI executed successfully.")
        logging.info("APPLICATION END")

    except Exception as e:
        logging.exception("CRITICAL ERROR: Failed to launch Multi-Agent Research & Report Generator application.")
        raise CustomException(e, sys)