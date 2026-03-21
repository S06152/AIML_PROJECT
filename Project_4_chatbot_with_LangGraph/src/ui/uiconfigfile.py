# Standard library imports
import sys
import os
from src.utils.logger import logging
from src.utils.exception import CustomException
from configparser import ConfigParser
from typing import List

class Config:
    """
    Centralized configuration manager.

    This class reads application settings from a `config.ini` file
    and provides typed getter methods to access configuration values.
    """

    def __init__(self, filename: str = "config.ini"):
        """
        Initialize the configuration manager.

        - Determines the absolute path of the config file.
        - Validates file existence.
        - Loads configuration values into memory.
        """
        try:
            logging.info("Initializing configuration manager.")

            # Get directory where this file exists
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Construct full path to config file
            self.config_path = os.path.join(current_dir, filename)

            logging.info(f"Reading configuration file from: {self.config_path}")

            # Ensure config file exists before reading
            if not os.path.exists(self.config_path):
                logging.error(f"Config file not found at path: {self.config_path}")
                raise FileNotFoundError(f"{filename} not found.")

            # Initialize ConfigParser and read file
            self.config = ConfigParser()
            self.config.read(self.config_path)

            logging.info("Configuration file loaded successfully.")

        except Exception as e:
            logging.exception("Error during Config initialization.")
            raise CustomException(e, sys)

    def get_page_title(self) -> str:
        """
        Returns the application page title.
        """
        try:
            value = self.config["DEFAULT"].get("PAGE_TITLE")
            logging.debug(f"PAGE_TITLE fetched: {value}")
            return value

        except Exception as e:
            logging.exception("Error fetching PAGE_TITLE.")
            raise CustomException(e, sys)

    def get_groq_model_options(self) -> List[str]:
        """
        Returns available Groq model options as a list of strings.
        Example format in config:
        GROQ_MODEL_OPTIONS = model1, model2, model3
        """
        try:
            options = self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS")

            # Convert comma-separated string into list
            parsed = [option.strip() for option in options.split(",") if option.strip()]

            logging.debug(f"GROQ_MODEL_OPTIONS fetched: {parsed}")
            return parsed
        
        except Exception as e:
            logging.exception("Error fetching GROQ_MODEL_OPTIONS.")
            raise CustomException(e, sys)