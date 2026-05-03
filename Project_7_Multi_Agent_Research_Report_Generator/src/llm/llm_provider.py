import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_groq import ChatGroq
from typing import Optional, Dict
import warnings
warnings.filterwarnings("ignore")

class LLMProvider():
    """
    Factory + Singleton provider for ChatGroq LLM.

    Responsibilities:
        - Validate user configuration
        - Initialize ChatGroq LLM
        - Cache and reuse LLM instance (singleton pattern)
        - Provide a clean interface for agents

    Expected Input (user_controls_input):
        {
            "GROQ_API_KEY": str,
            "LLM_MODEL": str,
            "TEMPERATURE": float,
            "TOKEN": int
        }
    """

    def __init__(self, user_controls_input: Dict) -> None:
        """
        Initialize LLMProvider with validated configuration.

        Args:
            user_controls_input (Dict): UI configuration

        Raises:
            CustomException: If initialization fails
        """

        try:
            logging.info("LLM PROVIDER INITIALIZATION START")

            # Extract configuration safely
            self._api_key: str = user_controls_input["GROQ_API_KEY"]
            self._model_name: str = user_controls_input["LLM_MODEL"]
            self._temperature: float = float(user_controls_input["TEMPERATURE"])
            self._max_tokens: int = int(user_controls_input["TOKEN"])
            
            # Validate configuration
            if not self._api_key:
                logging.error("GROQ_API_KEY is missing.")
                raise ValueError("GROQ_API_KEY is required.")

            if not self._model_name:
                logging.error("LLM_MODEL is missing.")
                raise ValueError("LLM model name is required.")
            
            # Instance-level LLM cache
            self._llm_instance: Optional[ChatGroq] = None

            logging.info(
                "LLMProvider configured successfully | Model = %s | Temp = %.2f | MaxTokens = %d",
                self._model_name,
                self._temperature,
                self._max_tokens
            )

            logging.info("LLM PROVIDER INITIALIZATION COMPLETE")

        except Exception as e:
            logging.exception("ERROR during LLMProvider initialization.")
            raise CustomException(e, sys)

    def get_llm(self)-> ChatGroq:
        """
        Returns cached ChatGroq instance (lazy initialization).

        Returns:
            ChatGroq: Configured LLM instance

        Raises:
            CustomException: If initialization fails
        """

        try:
            logging.info("LLM instance requested.")

            # Return cached instance if already created
            if self._llm_instance is not None:
                logging.info("Returning existing LLM instance (Singleton pattern).")
                return self._llm_instance

            # Create new LLM instance
            logging.info("Creating new ChatGroq instance | Model=%s", self._model_name)
            
            self._llm_instance = ChatGroq(
                model = self._model_name,
                api_key = self._api_key,
                temperature = self._temperature,
                max_tokens = self._max_tokens
            )

            logging.info("LLM instance created successfully")
            return self._llm_instance

        except Exception as e:
            logging.exception("Failed to create LLM instance")
            raise CustomException(e, sys)