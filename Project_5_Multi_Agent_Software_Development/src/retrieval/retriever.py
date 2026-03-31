# Standard library imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Optional
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
import warnings
warnings.filterwarnings("ignore")

_DEFAULT_TOP_K: int = 5
_SEARCH_TYPE: str = "similarity"

class Retriever:
    """
    Configures and caches a VectorStoreRetriever over the AUTOSAR SWS index.

    Design:
        - Lazy initialization: retriever is built on first call to get_retriever().
        - Singleton per instance: the same retriever object is reused on all calls.
        - Configurable top-K (number of chunks returned per query).

    Usage:
        retriever = Retriever(vector_store, top_k=5)
        langchain_retriever = retriever.get_retriever()
    """

    def __init__(self, vector_store: VectorStore, top_k: int = _DEFAULT_TOP_K) -> None:
        """
        Initialize Retriever with a vector store and retrieval configuration.

        Args:
            vector_store (VectorStore): Initialized LangChain vector store.
            top_k        (int)        : Number of top similar chunks to return per query (default: 5).

        Raises:
            Raises Exception: If vector_store is None.
        """
        try:
            logging.info("Initializing Retriever: top_k=%d, search_type='%s'.", top_k, _SEARCH_TYPE)

            if vector_store is None:
                raise ValueError("vector_store must not be None.")
        
            self._vector_store: VectorStore = vector_store
            self._top_k: int = top_k
            self._retriever: Optional[VectorStoreRetriever] = None

            logging.info("Retriever initialized")

        except Exception as e:
            logging.exception("Error during Retriever initialization.")
            raise CustomException(e, sys)

    def get_retriever(self) -> VectorStoreRetriever:
        """
        Return the configured VectorStoreRetriever (lazy singleton).

        On first call: builds and caches the retriever.
        On subsequent calls: returns the cached instance.

        Returns:
            VectorStoreRetriever: Configured retriever ready for use in chains.

        Raises:
            Raises Exception: If retriever creation fails.
        """

        try:
            # Lazy initialization
            if self._retriever is None:
                self._retriever = self._vector_store.as_retriever(search_type = _SEARCH_TYPE, search_kwargs = {"k": self._top_k})

                logging.info("VectorStoreRetriever created successfully.")

            else:
                logging.info("Using cached VectorStoreRetriever instance.")

            return self._retriever

        except Exception as e:
            logging.exception("Error while creating or retrieving VectorStoreRetriever.")
            raise CustomException(e, sys)