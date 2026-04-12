# Standard Library Imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.main import Load_Multi_Agent_Software_Development
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    try:
        logging.info("=" * 70)
        logging.info("AUTOSAR SWS Multi-Agent Software Development System — Starting")
        logging.info("=" * 70)

        pipeline = Load_Multi_Agent_Software_Development()
        logging.info("Pipeline instance created. Launching Streamlit application.")

        pipeline.run()

    except Exception as e:
        logging.exception("Fatal error occurred while launching the application.")
        raise CustomException(e, sys)

