# Standard library imports
import sys
import os
from src.utils.logger import logging
from src.utils.exception import CustomException
from configparser import ConfigParser
from typing import List
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Default Configuration File Name
# ---------------------------------------------------------------------------
_CONFIG_FILE_NAME: str = "config.ini"
_DEFAULT_SECTION: str = "DEFAULT"

class Config:
    """
    Centralized configuration manager.

    This class reads application settings from a `config.ini` file
    and provides typed getter methods to access configuration values.
    """

    def __init__(self, filename: str = _CONFIG_FILE_NAME) -> None:
        """
        Initialize and validate the configuration file.

        Args:
            filename (str): Config file name (must sit next to this module).

        Raises:
            Raises Exception: If file is not found or cannot be parsed.
        """

        try:
            logging.info("Initializing configuration manager.")

            # Get directory where this file exists
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Construct full path to config file
            self._config_path = os.path.join(current_dir, filename)

            logging.info(f"Reading configuration file from: {self._config_path}")

            # Ensure config file exists before reading
            if not os.path.exists(self._config_path):
                logging.error(f"Config file not found at path: {self._config_path}")
                raise FileNotFoundError(f"{filename} not found.")

            # Initialize ConfigParser and read file
            self._parser = ConfigParser()
            self._parser.read(self._config_path)

            logging.info("Configuration file loaded successfully.")

        except Exception as e:
            logging.exception("Error during Config initialization.")
            raise CustomException(e, sys)

    def get_page_title(self) -> str:
        """
        Returns the application page title.
        """
        try:
            value = self._parser[_DEFAULT_SECTION].get("PAGE_TITLE")
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
            options = self._parser[_DEFAULT_SECTION].get("GROQ_MODEL_OPTIONS")

            # Convert comma-separated string into list
            parsed = [option.strip() for option in options.split(",") if option.strip()]

            logging.debug(f"GROQ_MODEL_OPTIONS fetched: {parsed}")
            return parsed
        
        except Exception as e:
            logging.exception("Error fetching GROQ_MODEL_OPTIONS.")
            raise CustomException(e, sys)

    def get_temperature(self) -> List[float]:
        """
        Returns list of temperature values for LLM configuration.
        Converts string values into float.
        """
        try:
            options = self._parser[_DEFAULT_SECTION].get("TEMPERATURE")
            parsed = [float(option.strip()) for option in options.split(",") if option.strip()]

            logging.debug(f"TEMPERATURE fetched: {parsed}")
            return parsed

        except Exception as e:
            logging.exception("Error fetching TEMPERATURE.")
            raise CustomException(e, sys)

    def get_token(self) -> List[int]:
        """
        Returns list of token limits for LLM.
        Converts string values into integers.
        """
        try:
            options = self._parser[_DEFAULT_SECTION].get("TOKEN")
            parsed = [int(option.strip()) for option in options.split(",") if option.strip()]

            logging.debug(f"TOKEN fetched: {parsed}")
            return parsed

        except Exception as e:
            logging.exception("Error fetching TOKEN.")
            raise CustomException(e, sys)

    def get_chunk_size(self) -> int:
        """
        Returns chunk size for document splitting.
        """
        try:
            value = int(self._parser[_DEFAULT_SECTION].get("CHUNK_SIZE"))
            logging.debug(f"CHUNK_SIZE fetched: {value}")
            return value

        except Exception as e:
            logging.exception("Error fetching CHUNK_SIZE.")
            raise CustomException(e, sys)

    def get_chunk_overlap(self) -> int:
        """
        Returns chunk overlap size for document splitting.
        """
        try:
            value = int(self._parser[_DEFAULT_SECTION].get("CHUNK_OVERLAP"))
            logging.debug(f"CHUNK_OVERLAP fetched: {value}")
            return value

        except Exception as e:
            logging.exception("Error fetching CHUNK_OVERLAP.")
            raise CustomException(e, sys)

    def get_embedding_model(self) -> str:
        """
        Returns the embedding model name used for vectorization.
        """
        try:
            value = self._parser[_DEFAULT_SECTION].get("EMBEDDING_MODEL")
            logging.debug(f"EMBEDDING_MODEL fetched: {value}")
            return value

        except Exception as e:
            logging.exception("Error fetching EMBEDDING_MODEL.")
            raise CustomException(e, sys)
    
    def get_top_k(self) -> int:
        """
        Returns the number of top retrieved documents (Top-K) 
        used during similarity search.
        """
        try:
            value = int(self._parser[_DEFAULT_SECTION].get("TOP_K"))
            logging.debug(f"TOP_K fetched: {value}")
            return value

        except Exception as e:
            logging.exception("Error fetching TOP_K.")
            raise CustomException(e, sys)