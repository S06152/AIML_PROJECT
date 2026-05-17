# Standard Library Imports
import os
import sys
import uuid
from typing import List, Dict, Optional

# Third-Party Imports
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

# Custom Imports
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.embedding.embedding import EmbeddingManager

import streamlit as st


class PineconeVector:
    """
    Pinecone Vector Store Manager for Multi-Modal RAG Systems.

    Responsibilities:
        - Create Pinecone indexes
        - Store multi-modal embeddings
        - Manage vector metadata
        - Support hybrid retrieval
        - Enable semantic search across:
            * Text
            * Tables
            * Images
            * Charts

    Compatible With:
        - CLIP Embeddings
        - LangChain
        - Pinecone Serverless
    """

    def __init__(
        self,
        vector_records: List[Dict],
        index_name: str = "multi-modal-rag-system",
        cloud: str = "aws",
        region: str = "us-east-1",
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize Pinecone configuration.

        Args:
            vector_records (List[Dict]):
                Multi-modal vector records.

            index_name (str):
                Pinecone index name.

            cloud (str):
                Cloud provider.

            region (str):
                Deployment region.

            api_key (Optional[str]):
                Pinecone API key.
        """

        try:
            logging.info(
                "Initializing PineconeVector."
            )

            # ---------------------------------------------------
            # API KEY
            # ---------------------------------------------------
            self.api_key = (
                api_key
                or st.secrets.get(
                    "PINECONE_API_KEY"
                )
            )

            if not self.api_key:

                raise ValueError(
                    "PINECONE_API_KEY is missing."
                )

            # ---------------------------------------------------
            # CONFIGURATION
            # ---------------------------------------------------
            self.vector_records = vector_records

            self.index_name = index_name

            self.cloud = cloud
            self.region = region

            # ---------------------------------------------------
            # PINECONE CLIENT
            # ---------------------------------------------------
            self.pc = Pinecone(
                api_key=self.api_key
            )

            logging.info(
                f"PineconeVector initialized "
                f"(index={self.index_name})"
            )

        except Exception as e:
            logging.exception(
                "Failed to initialize PineconeVector."
            )
            raise CustomException(e, sys)

    # ==========================================================
    # ENSURE INDEX EXISTS
    # ==========================================================
    def ensure_index(
        self,
        dimension: int,
    ) -> None:
        """
        Create Pinecone index if it does not exist.

        Args:
            dimension (int):
                Embedding vector dimension.
        """

        try:
            logging.info(
                "Checking Pinecone index."
            )

            existing_indexes = [
                index_info["name"]
                for index_info in self.pc.list_indexes()
            ]

            # --------------------------------------------------
            # CREATE INDEX IF NEEDED
            # --------------------------------------------------
            if self.index_name not in existing_indexes:

                logging.info(
                    f"Creating Pinecone index: "
                    f"{self.index_name}"
                )

                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region,
                    ),
                )

                logging.info(
                    f"Index '{self.index_name}' "
                    f"created successfully."
                )

            else:

                logging.info(
                    f"Index '{self.index_name}' "
                    f"already exists."
                )

        except Exception as e:
            logging.exception(
                "Error while ensuring Pinecone index."
            )
            raise CustomException(e, sys)

    # ==========================================================
    # CREATE VECTOR STORE
    # ==========================================================
    def create_vectorstore(
        self,
    ) -> PineconeVectorStore:
        """
        Create Pinecone vector store and upload embeddings.

        Workflow:
            1. Detect embedding dimension
            2. Ensure Pinecone index exists
            3. Upload vectors + metadata
            4. Return LangChain vector store

        Returns:
            PineconeVectorStore:
                Ready-to-use vector store.
        """

        try:
            logging.info(
                "Creating Pinecone vector store."
            )

            # --------------------------------------------------
            # DETECT VECTOR DIMENSION
            # --------------------------------------------------
            if not self.vector_records:

                raise ValueError(
                    "No vector records provided."
                )

            sample_embedding = (
                self.vector_records[0][
                    "embedding"
                ]
            )

            dimension = len(sample_embedding)

            logging.info(
                f"Detected embedding dimension: "
                f"{dimension}"
            )

            # --------------------------------------------------
            # ENSURE INDEX EXISTS
            # --------------------------------------------------
            self.ensure_index(dimension)

            # --------------------------------------------------
            # CONNECT TO INDEX
            # --------------------------------------------------
            index = self.pc.Index(
                self.index_name
            )

            # --------------------------------------------------
            # PREPARE UPSERT PAYLOAD
            # --------------------------------------------------
            vectors_to_upsert = []

            documents = []

            for record in self.vector_records:

                vector_id = str(uuid.uuid4())

                embedding = record[
                    "embedding"
                ]

                text = record["text"]

                metadata = record.get(
                    "metadata",
                    {},
                )

                vectors_to_upsert.append(
                    (
                        vector_id,
                        embedding,
                        metadata,
                    )
                )

                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )

            logging.info(
                f"Prepared "
                f"{len(vectors_to_upsert)} "
                f"vectors for upload."
            )

            # --------------------------------------------------
            # UPSERT TO PINECONE
            # --------------------------------------------------
            index.upsert(
                vectors=vectors_to_upsert
            )

            logging.info(
                "Vectors uploaded successfully."
            )

            # --------------------------------------------------
            # SET ENV VARIABLE
            # --------------------------------------------------
            os.environ[
                "PINECONE_API_KEY"
            ] = self.api_key

            # --------------------------------------------------
            # CREATE LANGCHAIN VECTOR STORE
            # --------------------------------------------------
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=EmbeddingManager(),
                text_key="text",
            )

            logging.info(
                "LangChain PineconeVectorStore "
                "created successfully."
            )

            return vectorstore

        except Exception as e:
            logging.exception(
                "Error while creating vector store."
            )
            raise CustomException(e, sys)