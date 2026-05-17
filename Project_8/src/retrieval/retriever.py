# Standard Library Imports
import sys
from typing import Optional, List

# LangChain Imports
from langchain_core.documents import Document
from langchain_core.vectorstores import (
    VectorStore,
    VectorStoreRetriever,
)

# Custom Imports
from src.utils.logger import logging
from src.utils.exception import CustomException

import warnings

warnings.filterwarnings("ignore")


# ===========================================================
# DEFAULT CONFIGURATION
# ===========================================================
_DEFAULT_TOP_K: int = 5
_SEARCH_TYPE: str = "similarity"


class Retriever:
    """
    Multi-Modal Retriever for Hybrid RAG Systems.

    Responsibilities:
        - Configure VectorStoreRetriever
        - Retrieve relevant multi-modal context
        - Support:
            * Text retrieval
            * Table retrieval
            * Image retrieval
            * Chart retrieval
        - Enable hybrid semantic search

    Design:
        - Lazy initialization
        - Retriever caching
        - Configurable top-k retrieval
        - Metadata-aware retrieval

    Compatible With:
        - Pinecone
        - LangChain Vector Stores
        - CLIP Embeddings
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        """
        Initialize Retriever.

        Args:
            vector_store (VectorStore):
                Initialized vector database.

            top_k (int):
                Number of retrieved results.

        Raises:
            Exception:
                If vector_store is invalid.
        """

        try:
            logging.info(
                "Initializing Multi-Modal Retriever "
                f"(top_k={top_k}, "
                f"search_type={_SEARCH_TYPE})"
            )

            if vector_store is None:
                raise ValueError(
                    "vector_store must not be None."
                )

            self._vector_store = vector_store
            self._top_k = top_k

            # Lazy-loaded retriever
            self._retriever: Optional[
                VectorStoreRetriever
            ] = None

            logging.info(
                "Retriever initialized successfully."
            )

        except Exception as e:
            logging.exception(
                "Error during Retriever initialization."
            )
            raise CustomException(e, sys)

    def get_retriever(self) -> VectorStoreRetriever:
        """
        Return cached VectorStoreRetriever instance.

        Lazy Initialization:
            - Creates retriever on first call
            - Reuses cached retriever afterward

        Returns:
            VectorStoreRetriever:
                Configured retriever instance.
        """

        try:
            if self._retriever is None:

                logging.info(
                    "Creating VectorStoreRetriever instance."
                )

                self._retriever = (
                    self._vector_store.as_retriever(
                        search_type=_SEARCH_TYPE,
                        search_kwargs={
                            "k": self._top_k,
                        },
                    )
                )

                logging.info(
                    "VectorStoreRetriever created successfully."
                )

            else:

                logging.info(
                    "Using cached VectorStoreRetriever."
                )

            return self._retriever

        except Exception as e:
            logging.exception(
                "Failed to create/retrieve retriever."
            )
            raise CustomException(e, sys)

    def retrieve_multimodal_context(
        self,
        query: str,
    ) -> List[Document]:
        """
        Retrieve relevant multi-modal documents.

        Supported Retrieval Types:
            - Text chunks
            - Tables
            - Images
            - Charts/Figures

        Args:
            query (str):
                User query.

        Returns:
            List[Document]:
                Retrieved multi-modal documents.

        Example Queries:
            - "Summarize the paper"
            - "What does Figure 3 show?"
            - "Explain the chart on page 5"
            - "What trends are shown in the table?"
        """

        try:
            logging.info(
                f"Retrieving multi-modal context for query: "
                f"{query}"
            )

            retriever = self.get_retriever()

            retrieved_documents = (
                retriever.invoke(query)
            )

            logging.info(
                f"Retrieved "
                f"{len(retrieved_documents)} "
                f"documents successfully."
            )

            return retrieved_documents

        except Exception as e:
            logging.exception(
                "Error during multi-modal retrieval."
            )
            raise CustomException(e, sys)

    def retrieve_by_modality(
        self,
        query: str,
        modality: str,
    ) -> List[Document]:
        """
        Retrieve documents filtered by modality.

        Supported Modalities:
            - text
            - table
            - image

        Args:
            query (str):
                User query.

            modality (str):
                Desired modality type.

        Returns:
            List[Document]:
                Filtered retrieval results.
        """

        try:
            logging.info(
                f"Retrieving modality='{modality}' "
                f"for query='{query}'"
            )

            retriever = (
                self._vector_store.as_retriever(
                    search_type=_SEARCH_TYPE,
                    search_kwargs={
                        "k": self._top_k,
                        "filter": {
                            "modality": modality
                        },
                    },
                )
            )

            results = retriever.invoke(query)

            logging.info(
                f"Retrieved {len(results)} "
                f"{modality} documents."
            )

            return results

        except Exception as e:
            logging.exception(
                "Error during modality-specific retrieval."
            )
            raise CustomException(e, sys)