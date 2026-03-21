# Standard Library Imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore

class AstraDBVector:
    """
    Creates and manages an AstraDB vector store using LangChain.

    Responsibilities:
    - Validate AstraDB credentials
    - Initialize AstraDB vector store
    - Push documents + embeddings into AstraDB collection
    """

    def __init__(
        self,
        documents: List[Document],
        embeddings: HuggingFaceEmbeddings,
        collection_name: str = "rag_knowledge_assistant",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        """
        Initialize AstraDB vector store configuration.

        Args:
            documents (List[Document]): Chunked documents to store.
            embeddings (HuggingFaceEmbeddings): Embedding model instance.
            collection_name (str): AstraDB collection name.
            api_key (str): AstraDB application token.
            api_endpoint (str): AstraDB API endpoint URL.
        """

        try:
            logging.info("Initializing AstraDBVector configuration.")

            self.documents = documents
            self.embeddings = embeddings
            self.collection_name = collection_name
            self.api_key = api_key
            self.api_endpoint = api_endpoint

            # ---------------------------
            # Credential Validation
            # ---------------------------
            if not self.api_key:
                logging.error("ASTRA_DB_APPLICATION_TOKEN is missing.")
                raise ValueError(
                    "ASTRA_DB_APPLICATION_TOKEN is not set. "
                    "Provide it via Streamlit sidebar or environment variable."
                )

            if not self.api_endpoint:
                logging.error("ASTRA_DB_API_ENDPOINT is missing.")
                raise ValueError(
                    "ASTRA_DB_API_ENDPOINT is not set. "
                    "Provide it via Streamlit sidebar or environment variable."
                )

            logging.info(
                f"AstraDBVector initialized successfully with collection: {self.collection_name}"
            )

        except Exception as e:
            logging.exception("Error during AstraDBVector initialization.")
            raise CustomException(e, sys)

    def create_vectorstore(self) -> AstraDBVectorStore:
        """
        Create and return AstraDBVectorStore instance.

        This will:
        - Generate embeddings for documents
        - Upload vectors to AstraDB
        - Create collection if it doesn't exist

        Returns:
            AstraDBVectorStore: Initialized vector store instance.
        """

        try:
            logging.info(
                f"Creating AstraDB vector store with collection '{self.collection_name}'."
            )
            logging.info(f"Number of documents to index: {len(self.documents)}")

            vectorstore = AstraDBVectorStore.from_documents(
                documents = self.documents,
                embedding = self.embeddings,
                collection_name = self.collection_name,
                token = self.api_key,
                api_endpoint = self.api_endpoint,
            )

            logging.info("AstraDB vector store created and documents indexed successfully.")

            return vectorstore

        except Exception as e:
            logging.exception("Error while creating AstraDB vector store.")
            raise CustomException(e, sys)