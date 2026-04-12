"""
settings.py — Centralized configuration manager for AUTOSAR MAS.

Reads application settings from config.ini and exposes typed getter methods.
Follows the Single Responsibility Principle: only reads and returns config values.

MNC Standard: No magic strings inside business logic — all config is centralized here.
"""

import os
import sys
from configparser import ConfigParser
from typing import List
from src.utils.logger import logger
from src.utils.exception import CustomException

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CONFIG_FILE_NAME: str = "config.ini"
_DEFAULT_SECTION: str = "DEFAULT"


class Config:
    """
    Centralized configuration manager for the AUTOSAR MAS application.

    Reads settings from config.ini located in the same directory as this file.
    All getters return typed Python values (str, int, float, List).

    Usage:
        config = Config()
        title = config.get_page_title()
        models = config.get_groq_model_options()
    """

    def __init__(self, filename: str = _CONFIG_FILE_NAME) -> None:
        """
        Initialize Config and load the config.ini file.

        Args:
            filename (str): Config file name (must reside in the same folder).

        Raises:
            CustomException: If file is not found or cannot be parsed.
        """
        try:
            logger.info("Initializing Config manager.")

            current_dir: str = os.path.dirname(os.path.abspath(__file__))
            self._config_path: str = os.path.join(current_dir, filename)

            if not os.path.exists(self._config_path):
                raise FileNotFoundError(
                    f"Config file not found: {self._config_path}"
                )

            self._parser = ConfigParser()
            self._parser.read(self._config_path)

            logger.info("Config loaded from: %s", self._config_path)

        except Exception as e:
            raise CustomException(e, sys) from e

    # -----------------------------------------------------------------------
    # Getter Methods
    # -----------------------------------------------------------------------

    def get_page_title(self) -> str:
        """Return the Streamlit page title string."""
        try:
            return self._parser[_DEFAULT_SECTION].get("PAGE_TITLE", "AUTOSAR MAS")
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_groq_model_options(self) -> List[str]:
        """Return list of available Groq model identifiers."""
        try:
            raw: str = self._parser[_DEFAULT_SECTION].get(
                "GROQ_MODEL_OPTIONS", "llama-3.3-70b-versatile"
            )
            return [opt.strip() for opt in raw.split(",") if opt.strip()]
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_temperature(self) -> List[float]:
        """Return [min_temp, default_temp, max_temp] as floats."""
        try:
            raw: str = self._parser[_DEFAULT_SECTION].get("TEMPERATURE", "0.0, 0.2, 1.0")
            return [float(v.strip()) for v in raw.split(",") if v.strip()]
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_token(self) -> List[int]:
        """Return [min_tokens, default_tokens, max_tokens] as ints."""
        try:
            raw: str = self._parser[_DEFAULT_SECTION].get("TOKEN", "512, 2048, 8192")
            return [int(v.strip()) for v in raw.split(",") if v.strip()]
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_chunk_size(self) -> int:
        """Return document chunk size in characters."""
        try:
            return int(self._parser[_DEFAULT_SECTION].get("CHUNK_SIZE", "1200"))
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_chunk_overlap(self) -> int:
        """Return chunk overlap size in characters."""
        try:
            return int(self._parser[_DEFAULT_SECTION].get("CHUNK_OVERLAP", "200"))
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_embedding_model(self) -> str:
        """Return the HuggingFace embedding model identifier."""
        try:
            return self._parser[_DEFAULT_SECTION].get(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_top_k(self) -> int:
        """Return the number of top-K chunks retrieved per query."""
        try:
            return int(self._parser[_DEFAULT_SECTION].get("TOP_K", "5"))
        except Exception as e:
            raise CustomException(e, sys) from e
