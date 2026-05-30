# Standard library imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingManager:
    """
    Manages creation of HuggingFace embedding models.

    This class implements a lazy singleton pattern:
    - The embedding model is created only once.
    - Subsequent calls reuse the same instance.
    
    This improves performance and avoids reloading the model multiple times.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager.

        Args:
            model_name (str): HuggingFace embedding model name.
                              Default: lightweight MiniLM model.
        """
        try:
            logging.info(f"Initializing EmbeddingManager with model: {model_name}")

            # Store model name
            self.model_name = model_name

            # Placeholder for lazy initialization
            self._embeddings = None

        except Exception as e:
            logging.exception("Error during EmbeddingManager initialization.")
            raise CustomException(e, sys)

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Create and return HuggingFaceEmbeddings instance.

        Uses lazy initialization:
        - If embeddings are not created, instantiate them.
        - Otherwise, reuse the existing instance.

        Returns:
            HuggingFaceEmbeddings: Embedding model ready for vectorization.
        """
        try:
            # Create embeddings only once (singleton pattern)
            if self._embeddings is None:
                logging.info(
                    f"Creating HuggingFaceEmbeddings instance with model: {self.model_name}"
                )

                self._embeddings = HuggingFaceEmbeddings(
                    model_name = self.model_name
                )

                logging.info("HuggingFaceEmbeddings instance created successfully.")

            else:
                logging.info("Reusing existing HuggingFaceEmbeddings instance.")

            return self._embeddings

        except Exception as e:
            logging.exception("Error while creating HuggingFaceEmbeddings instance.")
            raise CustomException(e, sys)