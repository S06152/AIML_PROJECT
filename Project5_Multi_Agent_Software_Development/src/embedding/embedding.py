# Standard library imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Default embedding model — lightweight, fast, good semantic search quality
# ---------------------------------------------------------------------------
_DEFAULT_MODEL: str = "all-MiniLM-L6-v2"

class EmbeddingManager:
    """
    Singleton factory for HuggingFace sentence-transformer embedding models.

    The embedding model is loaded once on first call to ``create_embeddings()``
    and cached as a private attribute. Subsequent calls return the same instance
    without reloading the model weights (which can take several seconds).

    Usage:
        manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
        embeddings = manager.create_embeddings()
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        """
        Initialize EmbeddingManager with the target model name.

        Args:
            model_name (str): HuggingFace model identifier.
                              Default: 'all-MiniLM-L6-v2'.
        """
        try:
            logging.info(f"Initializing EmbeddingManager with model: {model_name}")

            # Store model name
            self._model_name = model_name

            # Placeholder for lazy initialization
            self._embeddings: Optional[HuggingFaceEmbeddings] = None

            logging.info("IEmbeddingManager ready")

        except Exception as e:
            logging.exception("Error during EmbeddingManager initialization.")
            raise CustomException(e, sys)

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return the HuggingFaceEmbeddings instance (lazy singleton).

        On first call: loads the model from HuggingFace Hub.
        On subsequent calls: returns the cached instance.

        Returns:
            HuggingFaceEmbeddings: Ready-to-use embedding model.

        Raises:
            AutosarMASException: If model loading fails.
        """

        try:
            # Create embeddings only once (singleton pattern)
            if self._embeddings is None:
                logging.info(
                    f"Creating HuggingFaceEmbeddings instance with model: {self.model_name}"
                )

                self._embeddings = HuggingFaceEmbeddings(
                    model_name = self._model_name
                )

                logging.info("HuggingFaceEmbeddings instance created successfully.")

            else:
                logging.info("Reusing existing HuggingFaceEmbeddings instance.")

            return self._embeddings

        except Exception as e:
            logging.exception("Error while creating HuggingFaceEmbeddings instance.")
            raise CustomException(e, sys)