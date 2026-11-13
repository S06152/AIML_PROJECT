import sys
from typing import List
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import warnings
warnings.filterwarnings("ignore")

class SparseVectorStore:
    """
    Creates and manages a BM25-based sparse retriever.

    Responsibilities:
        - Accept chunked documents.
        - Build a BM25 keyword-based index.
        - Retrieve relevant documents using keyword matching.
        - Return an initialized BM25 retriever.
    """

    def __init__(self, documents: List[Document], top_k : int = 5) -> None:
        """
        Initialize SparseVectorStore configuration.

        Args:
            documents (List[Document]): Chunked documents.
            top_k (int): Number of documents to retrieve.

        Raises:
            CustomException: If initialization fails.
        """

        try:
            logging.info("Initializing SparseVectorStore.")

            self.documents = documents
            self.top_k = top_k

            if not self.documents:
                raise ValueError("No documents provided for sparse indexing.")

            if self.top_k <= 0:
                raise ValueError("top_k must be greater than zero.")

            logging.info("SparseVectorStore initialized with %d documents.", len(self.documents))

        except Exception as e:
            logging.exception("Error during SparseVectorStore initialization.")
            raise CustomException(e, sys)

    def create_sparse_retriever(self):
        """
        Create and return a BM25 sparse retriever.

        BM25 uses keyword matching and term frequency
        to identify relevant documents.

        Returns:
            BM25Retriever: Initialized sparse retriever.

        Raises:
            CustomException: If retriever creation fails.
        """
        try:
            logging.info("Creating BM25 sparse retriever.")

            logging.info("Number of documents to index: %d", len(self.documents))

            sparse_retriever = BM25Retriever.from_documents(
                documents = self.documents,
                k = self.top_k
            )

            logging.info("BM25 sparse retriever created successfully.")

            return sparse_retriever

        except Exception as e:
            logging.exception("Error while creating BM25 sparse retriever.")
            raise CustomException(e, sys)