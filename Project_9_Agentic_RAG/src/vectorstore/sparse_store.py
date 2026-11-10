import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import warnings
warnings.filterwarnings("ignore")

class SparseVectorStore:
    """
    Creates and manages a Chroma vector store using LangChain.

    Responsibilities:
    - Accept chunked documents
    - Generate embeddings
    - Store vectors inside Chroma DB
    - Return initialized Chroma vector store instance
    """

    def __init__(self, documents: List[Document], top_k: int = 5):
        """
        Initialize ChromaVectorStore configuration.

        Args:
            documents (List[Document]): Chunked documents to embed and store.
            embeddings (HuggingFaceEmbeddings): Embedding model instance.
        """
        try:
            logging.info("Initializing ChromaVectorStore.")

            self.documents = documents
            self.top_k = top_k

            logging.info(
                f"ChromaVectorStore initialized with {len(self.documents)} documents."
            )

        except Exception as e:
            logging.exception("Error during ChromaVectorStore initialization.")
            raise CustomException(e, sys)

    def create_Sparsevectorstore(self):
        """
        Create and return a Chroma vector store instance.

        This method:
        - Generates embeddings for documents
        - Stores them inside Chroma
        - Returns a ready-to-use vector store

        Returns:
            Chroma: Initialized vector store.
        """
        try:
            logging.info("Creating Chroma vector store from documents.")
            logging.info(f"Number of documents to index: {len(self.documents)}")

            vectorstore = BM25Retriever.from_documents(
                documents = self.documents,
                k = self.top_k
            )

            logging.info("Chroma vector store created successfully.")

            return vectorstore

        except Exception as e:
            logging.exception("Error while creating Chroma vector store.")
            raise CustomException(e, sys)