# Standard library imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

class Retriever:
    """
    Wrapper class that converts a LangChain VectorStore
    into a configurable VectorStoreRetriever.

    Responsibilities:
    - Configure similarity search
    - Control top-K document retrieval
    - Lazily initialize retriever (singleton pattern)
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        """
        Initialize Retriever.

        Args:
            vector_store (VectorStore): Initialized LangChain vector store.
            top_k (int): Number of top similar documents to retrieve.
        """
        try:
            logging.info(
                f"Initializing Retriever with top_k = {top_k}."
            )

            self.vector_store = vector_store
            self.top_k = top_k
            self._retriever = None

        except Exception as e:
            logging.exception("Error during Retriever initialization.")
            raise CustomException(e, sys)

    def get_retriever(self) -> VectorStoreRetriever:
        """
        Create and return a VectorStoreRetriever instance.

        Uses similarity search with configurable top-K results.
        Implements lazy initialization so retriever is created only once.

        Returns:
            VectorStoreRetriever: Configured retriever instance.
        """
        try:
            # Lazy initialization
            if self._retriever is None:
                logging.info(
                    f"Creating VectorStoreRetriever with similarity search (top_k = {self.top_k})."
                )

                self._retriever = self.vector_store.as_retriever(
                    search_type = "similarity",
                    search_kwargs = {"k": self.top_k}
                )

                logging.info("VectorStoreRetriever created successfully.")

            else:
                logging.info("Using cached VectorStoreRetriever instance.")

            return self._retriever

        except Exception as e:
            logging.exception("Error while creating or retrieving VectorStoreRetriever.")
            raise CustomException(e, sys)