import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_groq import ChatGroq
from typing import Optional

class LLMProvider():
    """
    Factory and cache for the ChatGroq LLM instance.

    Responsibilities:
        - Validate required credentials (GROQ_API_KEY).
        - Validate required model configuration (model name).
        - Build and cache a ChatGroq instance (Singleton per config set).
        - Expose a clean get_llm() interface consumed by all agents.

    Args:
        user_controls_input (dict): Dictionary from the Streamlit UI containing:
            - "GROQ_API_KEY"    (str) : Groq cloud API key.
            - "LLM_Model_Name"  (str) : Groq model identifier.
            - "TEMPERATURE"     (float): Sampling temperature (0.0–1.0).
            - "TOKEN"           (int) : Maximum output tokens.
    """

    def __init__(self, user_controls_input: dict) -> None:
        """
        Initialize LLMProvider with validated user configuration.

        Args:
            user_controls_input (dict): UI-sourced LLM configuration.

        Raises:
            Raises Exception: If required keys are missing or invalid.
        """

        try:
            logging.info("Initializing LLMProvider")

            # Load configuration from Config class
            self._api_key: str = user_controls_input["GROQ_API_KEY"]
            self._model_name: str = user_controls_input["LLM_Model_Name"]
            self._temperature: float = float(user_controls_input["TEMPERATURE"])
            self._max_tokens: int = int(user_controls_input["TOKEN"])
            
            # Instance-level LLM cache
            self._llm_instance: Optional[ChatGroq] = None

            logging.info(
                "LLMProvider configured: model=%s, temperature=%.2f, max_tokens=%d",
                self._model_name, self._temperature, self._max_tokens,
            )

        except Exception as e:
            logging.error("Error occurred while initializing LLMProvider")
            raise CustomException(e, sys)

    def get_llm(self)-> ChatGroq:
        """
        Return the cached ChatGroq LLM instance (lazy singleton).

        On first call, builds the ChatGroq instance and caches it.
        Subsequent calls return the same instance without rebuilding.

        Returns:
            ChatGroq: Configured Groq LLM ready for inference.

        Raises:
            Raises Exception: If ChatGroq initialization fails.
        """

        try:
            logging.info("Request received to get LLM instance.")

            # Return cached instance if already created
            if self._llm_instance is not None:
                logging.info("Returning existing LLM instance (Singleton pattern).")
                return self._llm_instance
            
            # Validate API key
            if not self._api_key:
                logging.error("GROQ_API_KEY is missing. Cannot initialize LLM.")
                raise ValueError("GROQ_API_KEY not found in environment variables")

            # Validate Model Name
            if not self._model_name:
                logging.error("Model name is missing in configuration.")
                raise ValueError("Model name not provided.")
            
            logging.info("Creating new ChatGroq LLM instance: model=%s", self._model_name)

            # Initialize LLM
            self._llm_instance = ChatGroq(
                model = self._model_name,
                api_key = self._api_key,
                temperature = self._temperature,
                token = self._max_tokens

            )

            logging.info("ChatGroq LLM instance created and cached successfully.")

            return self._llm_instance

        except Exception as e:
            logging.error("Error occurred while creating ChatGroq LLM instance")
            raise CustomException(e, sys)