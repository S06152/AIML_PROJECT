# Standard Library Imports
import os
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

class PineconeVector:
    """
    Creates and manages a Pinecone vector store using LangChain.

    Responsibilities:
    - Validate Pinecone API key
    - Ensure index exists (create if needed)
    - Detect embedding dimension automatically
    - Upload documents with embeddings
    - Return a ready-to-use Pinecone vector store
    """

    def __init__(
        self,
        documents: List[Document],
        embeddings: HuggingFaceEmbeddings,
        index_name: str = "rag-knowledge-assistant",
        cloud: str = "aws",
        region: str = "us-east-1",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Pinecone vector store configuration.

        Args:
            documents (List[Document]): Chunked documents to store.
            embeddings (HuggingFaceEmbeddings): Embedding model instance.
            index_name (str): Pinecone index name.
            cloud (str): Cloud provider (aws/gcp/azure).
            region (str): Cloud region.
            api_key (str): Pinecone API key.
        """
        try:
            logging.info("Initializing PineconeVector configuration.")

            self.documents = documents
            self.embeddings = embeddings
            self.index_name = index_name
            self.cloud = cloud
            self.region = region
            self.api_key = api_key

            if not self.api_key:
                logging.error("Pinecone API key is missing.")
                raise ValueError(
                    "PINECONE_API_KEY is not set. "
                    "Provide it via Streamlit sidebar or environment variable."
                )

            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)

            logging.info(
                f"PineconeVector initialized successfully with index '{self.index_name}'."
            )

        except Exception as e:
            logging.exception("Error during PineconeVector initialization.")
            raise CustomException(e, sys)

    def ensure_index(self, dimension: int) -> None:
        """
        Create Pinecone index if it does not already exist.

        Args:
            dimension (int): Embedding vector dimension.
        """
        try:
            logging.info("Checking if Pinecone index exists.")

            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            logging.debug(f"Existing Pinecone indexes: {existing_indexes}")

            if self.index_name not in existing_indexes:
                logging.info(
                    f"Index '{self.index_name}' not found. Creating new index."
                )

                self.pc.create_index(
                    name = self.index_name,
                    dimension = dimension,
                    metric = "cosine",
                    spec = ServerlessSpec(
                        cloud = self.cloud,
                        region = self.region
                    ),
                )

                logging.info(
                    f"Pinecone index '{self.index_name}' created successfully."
                )
            else:
                logging.info(
                    f"Pinecone index '{self.index_name}' already exists."
                )

        except Exception as e:
            logging.exception("Error while ensuring Pinecone index existence.")
            raise CustomException(e, sys)

    def create_vectorstore(self) -> PineconeVectorStore:
        """
        Index all documents into Pinecone and return the vector store.

        Steps:
        1. Detect embedding dimension.
        2. Ensure index exists with correct dimension.
        3. Upload documents with embeddings.
        4. Return LangChain PineconeVectorStore instance.

        Returns:
            PineconeVectorStore: Ready-to-use vector store.
        """
        try:
            logging.info("Creating Pinecone vector store.")

            # ---------------------------
            # Detect embedding dimension
            # ---------------------------
            logging.info("Detecting embedding vector dimension.")
            sample_vector = self.embeddings.embed_query("dimension check")
            dimension = len(sample_vector)

            logging.info(f"Detected embedding dimension: {dimension}")
            logging.info(f"Number of documents to index: {len(self.documents)}")

            # ---------------------------
            # Ensure index exists
            # ---------------------------
            self.ensure_index(dimension)

            # ---------------------------
            # Set env variable so langchain_pinecone can find it
            # ---------------------------
            os.environ["PINECONE_API_KEY"] = self.api_key

            # ---------------------------
            # Upload documents
            # ---------------------------
            vectorstore = PineconeVectorStore.from_documents(
                documents = self.documents,
                embedding = self.embeddings,
                index_name = self.index_name,
                pinecone_api_key = self.api_key
            )

            logging.info(
                f"Documents successfully indexed into Pinecone index '{self.index_name}'."
            )

            return vectorstore

        except Exception as e:
            logging.exception("Error while creating Pinecone vector store.")
            raise CustomException(e, sys)