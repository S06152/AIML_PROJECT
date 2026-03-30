# Standard Library Imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class ChromaVectorStore:
    """
    Creates an in-memory Chroma vector store from chunked AUTOSAR SWS documents.

    Design:
        - Accepts pre-chunked Document objects and a configured embedding model.
        - Delegates vector computation and storage to Chroma.from_documents().
        - Returns a ready-to-query Chroma instance.

    Usage:
        store = ChromaVectorStore(chunks, embeddings)
        vector_db = store.create_vectorstore()
    """

    def __init__(self, documents: List[Document], embeddings: HuggingFaceEmbeddings) -> None:
        """
        Initialize ChromaVectorStore with documents and embedding model.

        Args:
            documents  (List[Document])       : Chunked Document objects to index.
            embeddings (HuggingFaceEmbeddings): Configured embedding model instance.

        Raises:
            Raises Exception: If inputs are invalid.
        """

        try:
            logging.info("Initializing ChromaVectorStore.")

            if not documents:
                raise ValueError("Document list must not be empty.")
            if embeddings is None:
                raise ValueError("Embeddings instance must not be None.")
            
            self._documents = documents
            self._embeddings = embeddings

            logging.info(
                f"ChromaVectorStore initialized with {len(self._documents)} documents."
            )

        except Exception as e:
            logging.exception("Error during ChromaVectorStore initialization.")
            raise CustomException(e, sys)

    def create_vectorstore(self) -> Chroma:
        """
        Build and return the Chroma vector store from the provided documents.

        Returns:
            Chroma: Populated vector store ready for similarity search.

        Raises:
            Raises Exception: If Chroma indexing fails.
        """

        try:
            logging.info("Creating Chroma vector store from documents.")
            logging.info(f"Number of documents to index: {len(self._documents)}")

            vectorstore: Chroma = Chroma.from_documents(
                documents = self._documents,
                embedding = self._embeddings
            )

            logging.info("Chroma vector store created successfully.")

            return vectorstore

        except Exception as e:
            logging.exception("Error while creating Chroma vector store.")
            raise CustomException(e, sys)