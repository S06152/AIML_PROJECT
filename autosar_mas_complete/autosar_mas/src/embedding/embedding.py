"""
embedding.py — HuggingFace embedding model manager for AUTOSAR MAS.

Provides a singleton HuggingFaceEmbeddings instance used by ChromaVectorStore
to convert text chunks into 384-dimensional dense vectors.

Model choice: all-MiniLM-L6-v2
    - Lightweight (80MB), fast CPU inference.
    - Strong semantic search quality for technical English text.
    - No GPU required — suitable for local development.
"""

import sys
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logger import logger
from src.utils.exception import CustomException

# ---------------------------------------------------------------------------
# Default model — can be overridden via config.ini
# ---------------------------------------------------------------------------
_DEFAULT_MODEL: str = "all-MiniLM-L6-v2"


class EmbeddingManager:
    """
    Singleton factory for HuggingFace sentence-transformer embedding models.

    The model is loaded once on first call to create_embeddings()
    and cached. Subsequent calls return the same instance without reloading.

    Usage:
        manager    = EmbeddingManager(model_name="all-MiniLM-L6-v2")
        embeddings = manager.create_embeddings()
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        """
        Initialize EmbeddingManager with the target HuggingFace model name.

        Args:
            model_name (str): HuggingFace model identifier. Default: all-MiniLM-L6-v2.
        """
        try:
            logger.info("Initializing EmbeddingManager: model='%s'.", model_name)
            self._model_name: str = model_name
            self._embeddings: Optional[HuggingFaceEmbeddings] = None
            logger.info("EmbeddingManager ready.")

        except Exception as e:
            raise CustomException(e, sys) from e

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return the HuggingFaceEmbeddings instance (lazy singleton).

        On first call: downloads and loads the model (~30 seconds first time).
        On subsequent calls: returns the cached instance immediately.

        Returns:
            HuggingFaceEmbeddings: Configured embedding model ready for use.

        Raises:
            CustomException: If model loading fails.
        """
        try:
            if self._embeddings is None:
                logger.info(
                    "Loading HuggingFace model: '%s'. This may take ~30s on first run.",
                    self._model_name,
                )
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self._model_name
                )
                logger.info("HuggingFaceEmbeddings loaded and cached.")
            else:
                logger.info("Reusing cached HuggingFaceEmbeddings instance.")

            return self._embeddings

        except Exception as e:
            raise CustomException(e, sys) from e
