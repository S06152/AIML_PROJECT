import sys
import os
import shutil
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import warnings
warnings.filterwarnings("ignore")

class ChromaVectorStore:
    """
    Creates and manages a persistent Chroma vector store using LangChain.

    Responsibilities:
    - Accept chunked documents
    - Generate embeddings
    - Store vectors inside Chroma DB with disk persistence
    - Load existing vector store from persisted directory
    - Return initialized Chroma vector store instance
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        embeddings: HuggingFaceEmbeddings = None,
        persist_directory: str = "./chroma_db",
        collection_name: str = "agentic_rag_collection"
    ):
        """
        Initialize ChromaVectorStore configuration.

        Args:
            documents (Optional[List[Document]]): Chunked documents to embed and store.
            embeddings (HuggingFaceEmbeddings): Embedding model instance.
            persist_directory (str): Path to persist the vector store on disk.
            collection_name (str): Name of the Chroma collection.
        """
        try:
            logging.info("Initializing ChromaVectorStore.")

            self.documents = documents
            self.embeddings = embeddings
            self.persist_directory = os.path.abspath(persist_directory)
            self.collection_name = collection_name

            # Ensure persist directory exists
            os.makedirs(self.persist_directory, exist_ok=True)

            doc_count = len(self.documents) if self.documents else 0
            logging.info(
                f"ChromaVectorStore initialized | Documents: {doc_count} | "
                f"Persist Directory: {self.persist_directory} | "
                f"Collection: {self.collection_name}"
            )

        except Exception as e:
            logging.exception("Error during ChromaVectorStore initialization.")
            raise CustomException(e, sys)

    def create_vectorstore(self) -> Chroma:
        """
        Create and return a persistent Chroma vector store instance.

        This method:
        - Generates embeddings for documents
        - Stores them inside Chroma with disk persistence
        - Returns a ready-to-use vector store

        Returns:
            Chroma: Initialized and persisted vector store.
        """
        try:
            logging.info("Creating persistent Chroma vector store from documents.")
            logging.info(f"Number of documents to index: {len(self.documents)}")
            logging.info(f"Persist directory: {self.persist_directory}")
            logging.info(f"Collection name: {self.collection_name}")

            vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )

            logging.info("Persistent Chroma vector store created and saved successfully.")

            return vectorstore

        except Exception as e:
            logging.exception("Error while creating persistent Chroma vector store.")
            raise CustomException(e, sys)

    @classmethod
    def load_vectorstore(
        cls,
        embeddings: HuggingFaceEmbeddings,
        persist_directory: str = "./chroma_db",
        collection_name: str = "agentic_rag_collection"
    ) -> Optional[Chroma]:
        """
        Load an existing Chroma vector store from the persisted directory.

        Args:
            embeddings (HuggingFaceEmbeddings): Embedding model instance.
            persist_directory (str): Path to the persisted vector store.
            collection_name (str): Name of the Chroma collection.

        Returns:
            Optional[Chroma]: Loaded vector store, or None if not found.
        """
        try:
            abs_path = os.path.abspath(persist_directory)

            if not os.path.exists(abs_path):
                logging.info(f"No persisted vector store found at: {abs_path}")
                return None

            # Check if directory has actual data (not just empty folder)
            if not os.listdir(abs_path):
                logging.info(f"Persist directory is empty: {abs_path}")
                return None

            logging.info(f"Loading persisted Chroma vector store from: {abs_path}")
            logging.info(f"Collection name: {collection_name}")

            vectorstore = Chroma(
                persist_directory=abs_path,
                embedding_function=embeddings,
                collection_name=collection_name
            )

            # Verify the collection has documents
            collection_count = vectorstore._collection.count()

            if collection_count == 0:
                logging.info("Persisted collection exists but contains no documents.")
                return None

            logging.info(
                f"Persisted vector store loaded successfully | "
                f"Documents in collection: {collection_count}"
            )

            return vectorstore

        except Exception as e:
            logging.exception("Error while loading persisted Chroma vector store.")
            return None

    @classmethod
    def delete_vectorstore(cls, persist_directory: str = "./chroma_db") -> bool:
        """
        Delete the persisted vector store from disk.

        Args:
            persist_directory (str): Path to the persisted vector store.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            abs_path = os.path.abspath(persist_directory)

            if os.path.exists(abs_path):
                shutil.rmtree(abs_path)
                logging.info(f"Persisted vector store deleted: {abs_path}")
                return True
            else:
                logging.info(f"No persisted vector store to delete at: {abs_path}")
                return False

        except Exception as e:
            logging.exception("Error while deleting persisted vector store.")
            return False

    @staticmethod
    def store_exists(persist_directory: str = "./chroma_db") -> bool:
        """
        Check whether a persisted vector store exists and contains data.

        Args:
            persist_directory (str): Path to the persisted vector store.

        Returns:
            bool: True if store exists with data, False otherwise.
        """
        abs_path = os.path.abspath(persist_directory)
        return os.path.exists(abs_path) and len(os.listdir(abs_path)) > 0