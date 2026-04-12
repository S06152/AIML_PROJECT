"""
llm_provider.py — Singleton factory for the ChatGroq LLM instance.

Centralizes all LLM configuration validation and instantiation.
Used by all agents and the QA chain to obtain a shared LLM instance.

FIX vs original:
    - Corrected config key from "LLM_Model_Name" → "LLM_MODEL" (matches UI dict)
    - Corrected ChatGroq param: 'token' → 'max_tokens'
    - Added API key validation guard before LLM construction
"""

import sys
from typing import Optional
from langchain_groq import ChatGroq
from src.utils.logger import logger
from src.utils.exception import CustomException


class LLMProvider:
    """
    Factory and lazy singleton for the ChatGroq LLM instance.

    Validates all required credentials and configuration before constructing
    the LLM. Subsequent calls to get_llm() return the cached instance.

    Args:
        user_controls_input (dict): UI-sourced configuration containing:
            - "GROQ_API_KEY" (str)   : Groq Cloud API key.
            - "LLM_MODEL"    (str)   : Groq model identifier.
            - "TEMPERATURE"  (float) : Sampling temperature (0.0–1.0).
            - "TOKEN"        (int)   : Maximum output tokens.
    """

    def __init__(self, user_controls_input: dict) -> None:
        """
        Initialize LLMProvider with validated user configuration.

        Raises:
            CustomException: If required keys are missing or values are invalid.
        """
        try:
            logger.info("Initializing LLMProvider.")

            # ── Extract config values ────────────────────────────────────────
            self._api_key: str    = user_controls_input.get("GROQ_API_KEY", "")
            self._model_name: str = user_controls_input.get("LLM_MODEL", "")
            self._temperature: float = float(user_controls_input.get("TEMPERATURE", 0.2))
            self._max_tokens: int    = int(user_controls_input.get("TOKEN", 2048))

            # Lazy-init cache
            self._llm_instance: Optional[ChatGroq] = None

            logger.info(
                "LLMProvider configured: model=%s, temperature=%.2f, max_tokens=%d",
                self._model_name, self._temperature, self._max_tokens,
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_llm(self) -> ChatGroq:
        """
        Return the cached ChatGroq LLM instance (lazy singleton).

        Returns:
            ChatGroq: Configured and ready Groq LLM instance.

        Raises:
            CustomException: If API key or model name is missing, or init fails.
        """
        try:
            if self._llm_instance is not None:
                logger.info("Returning cached ChatGroq LLM instance.")
                return self._llm_instance

            # ── Validate required config ─────────────────────────────────────
            if not self._api_key:
                raise ValueError(
                    "GROQ_API_KEY is missing. "
                    "Set it in Streamlit secrets or .env file."
                )
            if not self._model_name:
                raise ValueError("LLM model name must not be empty.")

            logger.info("Creating ChatGroq instance: model=%s", self._model_name)

            # ── Build LLM ────────────────────────────────────────────────────
            # FIX: ChatGroq uses 'max_tokens', NOT 'token'
            self._llm_instance = ChatGroq(
                model=self._model_name,
                api_key=self._api_key,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            logger.info("ChatGroq instance created and cached successfully.")
            return self._llm_instance

        except Exception as e:
            raise CustomException(e, sys) from e
