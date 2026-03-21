# Standard Library Imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
import weaviate
from weaviate.classes.init import Auth

class WeaviateVector:
    """
    Creates and manages a Weaviate vector store using LangChain.

    Supports connection modes:
        1️⃣ Weaviate Cloud (WCS) — requires api_key + weaviate_url
        2️⃣ Custom URL instance
        3️⃣ Local Docker instance (default fallback)

    Responsibilities:
    - Initialize Weaviate client
    - Index documents
    - Load existing collection
    - Cleanly close client connection
    """

    def __init__(
        self,
        documents: List[Document],
        embeddings: HuggingFaceEmbeddings,
        index_name: str = "RAGKnowledgeAssistant",
        api_key: Optional[str] = None,
        weaviate_url: Optional[str] = None,
    ):
        """
        Initialize Weaviate configuration.

        Priority order for credentials:
            1. Explicitly passed values
            2. Environment variables
            3. Local fallback
        """
        try:
            logging.info("Initializing WeaviateVector configuration.")

            self.documents = documents
            self.embeddings = embeddings
            self.index_name = index_name

            # Credential resolution
            self.api_key = api_key 
            self.weaviate_url = weaviate_url

            logging.info(f"Using index name: {self.index_name}")

            if self.weaviate_url:
                logging.info(f"Weaviate URL detected: {self.weaviate_url}")
            else:
                logging.info("No Weaviate URL provided. Defaulting to local instance.")

            # Create Weaviate client
            self.client = self._create_client()

            logging.info("Weaviate client initialized successfully.")

        except Exception as e:
            logging.exception("Error during WeaviateVector initialization.")
            raise CustomException(e, sys)

    def _create_client(self) -> weaviate.WeaviateClient:
        """
        Create a Weaviate client connection.

        Returns:
            WeaviateClient instance.
        """
        try:
            if self.weaviate_url and self.api_key:
                logging.info("Connecting to Weaviate Cloud (WCS).")

                client = weaviate.connect_to_weaviate_cloud(
                    cluster_url = self.weaviate_url,
                    auth_credentials = Auth.api_key(self.api_key),
                )

            elif self.weaviate_url:
                logging.info("Connecting to custom Weaviate instance (no auth).")

                client = weaviate.connect_to_custom(
                    http_host = self.weaviate_url,
                    http_port = 8080,
                    grpc_port = 50051,
                    http_secure = False,
                    grpc_secure = False,
                )

            else:
                logging.info("Connecting to local Weaviate instance.")
                client = weaviate.connect_to_local()

            return client

        except Exception as e:
            logging.exception("Failed to create Weaviate client.")
            raise CustomException(e, sys)

    def create_vectorstore(self) -> WeaviateVectorStore:
        """
        Index documents into Weaviate and return vector store.

        Returns:
            WeaviateVectorStore: LangChain-compatible vector store.
        """
        try:
            logging.info("Creating Weaviate vector store.")
            logging.info(f"Number of documents to index: {len(self.documents)}")

            vectorstore = WeaviateVectorStore.from_documents(
                documents = self.documents,
                embedding = self.embeddings,
                client = self.client,
                index_name = self.index_name,
                text_key = "text",
            )

            logging.info(
                f"Documents successfully indexed into Weaviate collection '{self.index_name}'."
            )

            return vectorstore

        except Exception as e:
            logging.exception("Error while creating Weaviate vector store.")
            raise CustomException(e, sys)

    def load_existing_vectorstore(self) -> WeaviateVectorStore:
        """
        Connect to an already existing Weaviate collection.

        Returns:
            WeaviateVectorStore: Ready-to-use vector store.
        """
        try:
            logging.info(
                f"Loading existing Weaviate collection '{self.index_name}'."
            )

            return WeaviateVectorStore(
                client = self.client,
                index_name = self.index_name,
                text_key = "text",
                embedding = self.embeddings,
            )

        except Exception as e:
            logging.exception("Error while loading existing Weaviate collection.")
            raise CustomException(e, sys)

    def close(self):
        """
        Close the Weaviate client connection safely.
        """
        try:
            if self.client:
                logging.info("Closing Weaviate client connection.")
                self.client.close()
                logging.info("Weaviate client closed successfully.")

        except Exception as e:
            logging.exception("Error while closing Weaviate client.")
            raise CustomException(e, sys)