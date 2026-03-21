# Standard Library Imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.chroma_store import ChromaVectorStore
from src.vectorstore.pinecone_store import PineconeVector
from src.vectorstore.astradb_store import AstraDBVector
from src.vectorstore.weaviate_store import WeaviateVector

class VectorStoreFactory:
    """
    Factory class responsible for returning the correct vector store
    implementation based on user selection.

    Supports multiple vector databases:
    - FAISS (local in-memory)
    - Chroma (local persistent/in-memory)
    - Pinecone (cloud serverless)
    - AstraDB (DataStax cloud)
    - Weaviate (cloud/self-hosted)

    This enables:
    - Clean abstraction layer
    - Easy database switching
    - Centralized validation
    """

    # Mapping of vector DB names to their corresponding classes
    VECTOR_STORE_MAP = {
        "faiss": FaissVectorStore,
        "chroma": ChromaVectorStore,
        "pinecone": PineconeVector,
        "astradb": AstraDBVector,
        "weaviate": WeaviateVector,
    }

    @staticmethod
    def get_vector_store(
        vector_db_name: str,
        documents: List[Document],
        embeddings: HuggingFaceEmbeddings,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        """
        Return an instance of the selected vector store.

        Args:
            vector_db_name (str): Name of vector DB selected by user.
            documents (List[Document]): Chunked documents.
            embeddings (HuggingFaceEmbeddings): Embedding model.
            api_key (Optional[str]): API key for cloud vector DBs.
            api_endpoint (Optional[str]): Endpoint/URL if required.

        Returns:
            Instance of selected vector store class.

        Raises:
            CustomException: If unsupported vector DB or runtime error occurs.
        """
        try:
            logging.info(f"Vector store selection requested: {vector_db_name}")

            if not vector_db_name:
                logging.error("Vector DB name not provided.")
                raise ValueError("Vector database name must be provided.")

            vector_db_name = vector_db_name.lower()

            # Retrieve vector store class
            vector_db_class = VectorStoreFactory.VECTOR_STORE_MAP.get(vector_db_name)

            if vector_db_class is None:
                supported = list(VectorStoreFactory.VECTOR_STORE_MAP.keys())
                logging.error(
                    f"Unsupported vector database requested: {vector_db_name}"
                )
                raise ValueError(
                    f"Unsupported vector database: {vector_db_name}. "
                    f"Supported: {supported}"
                )

            logging.info(f"Vector store class selected: {vector_db_class.__name__}")
            logging.info(f"Number of documents received: {len(documents)}")

            # ---------------------------------------
            # Pass credentials to cloud-based stores
            # ---------------------------------------
            if vector_db_name == "pinecone":
                logging.info("Initializing Pinecone vector store.")
                return vector_db_class(
                    documents,
                    embeddings,
                    api_key = api_key
                )

            elif vector_db_name == "astradb":
                logging.info("Initializing AstraDB vector store.")
                return vector_db_class(
                    documents,
                    embeddings,
                    api_key = api_key,
                    api_endpoint = api_endpoint
                )

            elif vector_db_name == "weaviate":
                logging.info("Initializing Weaviate vector store.")
                return vector_db_class(
                    documents,
                    embeddings,
                    api_key = api_key,
                    weaviate_url = api_endpoint
                )

            # ---------------------------------------
            # Local vector stores (no credentials)
            # ---------------------------------------
            logging.info(f"Initializing local vector store: {vector_db_name}")
            return vector_db_class(documents, embeddings)

        except Exception as e:
            logging.exception("Error while selecting vector store.")
            raise CustomException(e, sys)