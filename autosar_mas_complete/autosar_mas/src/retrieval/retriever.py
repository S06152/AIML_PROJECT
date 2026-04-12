"""
retriever.py — VectorStore retriever configuration for AUTOSAR SWS RAG.

Wraps the Chroma vector store in a LangChain VectorStoreRetriever
with configurable top-K similarity search.

Design:
    - Lazy singleton: retriever is built on first call, cached thereafter.
    - Configurable search type and top-K.
    - Decoupled from the vector store implementation.
"""

import sys
from typing import Optional
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from src.utils.logger import logger
from src.utils.exception import CustomException

# ---------------------------------------------------------------------------
# Retriever defaults
# ---------------------------------------------------------------------------
_DEFAULT_TOP_K: int  = 5
_SEARCH_TYPE: str    = "similarity"


class Retriever:
    """
    Configures and caches a VectorStoreRetriever over the AUTOSAR SWS index.

    Usage:
        retriever_mgr       = Retriever(vector_store, top_k=5)
        langchain_retriever = retriever_mgr.get_retriever()
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        """
        Initialize Retriever with a vector store and search configuration.

        Args:
            vector_store (VectorStore): Initialized LangChain Chroma vector store.
            top_k        (int)        : Number of top similar chunks per query.

        Raises:
            CustomException: If vector_store is None.
        """
        try:
            logger.info(
                "Initializing Retriever: top_k=%d, search_type='%s'.",
                top_k, _SEARCH_TYPE,
            )

            if vector_store is None:
                raise ValueError("vector_store must not be None.")

            self._vector_store: VectorStore                    = vector_store
            self._top_k: int                                   = top_k
            self._retriever: Optional[VectorStoreRetriever]    = None

            logger.info("Retriever initialized.")

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_retriever(self) -> VectorStoreRetriever:
        """
        Return the configured VectorStoreRetriever (lazy singleton).

        Returns:
            VectorStoreRetriever: Ready-to-use retriever for LangChain chains.

        Raises:
            CustomException: If retriever creation fails.
        """
        try:
            if self._retriever is None:
                logger.info("Creating VectorStoreRetriever.")
                self._retriever = self._vector_store.as_retriever(
                    search_type=_SEARCH_TYPE,
                    search_kwargs={"k": self._top_k},
                )
                logger.info("VectorStoreRetriever created and cached.")
            else:
                logger.info("Reusing cached VectorStoreRetriever.")

            return self._retriever

        except Exception as e:
            raise CustomException(e, sys) from e
