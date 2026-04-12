"""
chroma_store.py — Chroma vector store for AUTOSAR SWS document chunks.

Indexes chunked AUTOSAR SWS Documents into an in-memory Chroma vector store
for fast similarity search during the RAG pipeline.

Design:
    - In-memory store: no persistence between sessions (fresh index per upload).
    - Delegates all vector math to Chroma.from_documents().
    - Single Responsibility: only creates and returns the vector store.
"""

import sys
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.utils.logger import logger
from src.utils.exception import CustomException


class ChromaVectorStore:
    """
    Creates an in-memory Chroma vector store from chunked AUTOSAR SWS documents.

    Usage:
        store      = ChromaVectorStore(chunks, embeddings)
        vector_db  = store.create_vectorstore()
    """

    def __init__(
        self,
        documents: List[Document],
        embeddings: HuggingFaceEmbeddings,
    ) -> None:
        """
        Initialize ChromaVectorStore with documents and embedding model.

        Args:
            documents  (List[Document])       : Chunked Document objects.
            embeddings (HuggingFaceEmbeddings): Configured embedding model.

        Raises:
            CustomException: If inputs are invalid.
        """
        try:
            logger.info("Initializing ChromaVectorStore.")

            if not documents:
                raise ValueError("Document list must not be empty.")
            if embeddings is None:
                raise ValueError("Embeddings instance must not be None.")

            self._documents: List[Document]        = documents
            self._embeddings: HuggingFaceEmbeddings = embeddings

            logger.info(
                "ChromaVectorStore ready: %d documents to index.", len(documents)
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def create_vectorstore(self) -> Chroma:
        """
        Build and return the Chroma vector store from the provided documents.

        Returns:
            Chroma: Populated vector store ready for similarity search queries.

        Raises:
            CustomException: If Chroma indexing fails.
        """
        try:
            logger.info(
                "Building Chroma vector store from %d documents.", len(self._documents)
            )

            vectorstore: Chroma = Chroma.from_documents(
                documents=self._documents,
                embedding=self._embeddings,
            )

            logger.info("Chroma vector store created successfully.")
            return vectorstore

        except Exception as e:
            raise CustomException(e, sys) from e
