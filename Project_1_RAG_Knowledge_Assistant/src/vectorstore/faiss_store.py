# Standard Library Imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class FaissVectorStore:
    """
    Creates and manages a FAISS vector store using LangChain.

    Responsibilities:
    - Accept chunked documents
    - Generate embeddings
    - Store vectors in FAISS (in-memory vector index)
    - Return initialized FAISS vector store instance
    """

    def __init__(self, documents: List[Document], embeddings: HuggingFaceEmbeddings):
        """
        Initialize FAISS vector store configuration.

        Args:
            documents (List[Document]): Chunked documents to embed and store.
            embeddings (HuggingFaceEmbeddings): Embedding model instance.
        """
        try:
            logging.info("Initializing FaissVectorStore.")

            self.documents = documents
            self.embeddings = embeddings

            logging.info(
                f"FaissVectorStore initialized with {len(self.documents)} documents."
            )

        except Exception as e:
            logging.exception("Error during FaissVectorStore initialization.")
            raise CustomException(e, sys)

    def create_vectorstore(self) -> FAISS:
        """
        Create and return a FAISS vector store instance.

        This method:
        - Generates embeddings for documents
        - Builds FAISS index in memory
        - Returns a ready-to-use vector store

        Returns:
            FAISS: Initialized FAISS vector store.
        """
        try:
            logging.info("Creating FAISS vector store from documents.")
            logging.info(f"Number of documents to index: {len(self.documents)}")

            vectorstore = FAISS.from_documents(
                documents = self.documents,
                embedding = self.embeddings
            )

            logging.info("FAISS vector store created successfully.")

            return vectorstore

        except Exception as e:
            logging.exception("Error while creating FAISS vector store.")
            raise CustomException(e, sys)