# Standard Library Imports
import sys
import io
import base64
from typing import List, Tuple

# Third-Party Imports
from PIL import Image

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Custom Imports
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.embedding.embedding import EmbeddingManager


class ChunkingStrategy:
    """
    Multi-Modal Chunking Strategy for RAG Systems.

    Responsibilities:
        - Split text into semantic chunks
        - Process table documents
        - Process image documents
        - Generate multi-modal embeddings
        - Prepare data for hybrid retrieval

    Supported Modalities:
        - Text
        - Tables
        - Images / Charts
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """
        Initialize chunking configuration.

        Args:
            chunk_size (int):
                Maximum size of each chunk.

            chunk_overlap (int):
                Overlapping characters between chunks
                to preserve contextual continuity.
        """

        try:
            logging.info(
                "Initializing ChunkingStrategy "
                f"(chunk_size={chunk_size}, "
                f"chunk_overlap={chunk_overlap})"
            )

            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

            # Text splitter for semantic chunking
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
            )

            logging.info(
                "RecursiveCharacterTextSplitter initialized successfully."
            )

        except Exception as e:
            logging.exception(
                "Failed to initialize ChunkingStrategy."
            )
            raise CustomException(e, sys)

    def split_documents_into_chunks(
        self,
        documents: List[Document],
    ) -> Tuple[List[Document], List]:
        """
        Process multi-modal documents and generate embeddings.

        Workflow:
            1. Chunk text documents
            2. Process table documents
            3. Process image documents
            4. Generate embeddings
            5. Prepare vector-store-ready objects

        Args:
            documents (List[Document]):
                Multi-modal LangChain documents.

        Returns:
            Tuple[List[Document], List]:
                - Chunked documents
                - Corresponding embeddings
        """

        all_documents = []
        all_embeddings = []

        try:
            logging.info(
                "Starting multi-modal chunking pipeline."
            )

            for index, document in enumerate(documents):

                modality = document.metadata.get(
                    "modality",
                    "text",
                )

                # ===================================================
                # TEXT DOCUMENT PROCESSING
                # ===================================================
                if modality == "text":

                    logging.info(
                        f"Processing text document: {index}"
                    )

                    text_chunks = self.splitter.split_documents(
                        [document]
                    )

                    for chunk in text_chunks:

                        # Generate text embedding
                        embedding = (
                            EmbeddingManager.embed_text(
                                chunk.page_content
                            )
                        )

                        chunk.metadata["embedding_type"] = (
                            "text_embedding"
                        )

                        all_documents.append(chunk)
                        all_embeddings.append(embedding)

                # ===================================================
                # TABLE DOCUMENT PROCESSING
                # ===================================================
                elif modality == "table":

                    logging.info(
                        f"Processing table document: {index}"
                    )

                    table_content = document.page_content

                    # Tables can also be chunked if large
                    table_chunks = self.splitter.split_documents(
                        [
                            Document(
                                page_content=table_content,
                                metadata=document.metadata,
                            )
                        ]
                    )

                    for chunk in table_chunks:

                        embedding = (
                            EmbeddingManager.embed_text(
                                chunk.page_content
                            )
                        )

                        chunk.metadata["embedding_type"] = (
                            "table_embedding"
                        )

                        all_documents.append(chunk)
                        all_embeddings.append(embedding)

                # ===================================================
                # IMAGE / CHART DOCUMENT PROCESSING
                # ===================================================
                elif modality == "image":

                    logging.info(
                        f"Processing image document: {index}"
                    )

                    image_base64 = document.metadata.get(
                        "image_base64"
                    )

                    if not image_base64:

                        logging.warning(
                            "No image_base64 found in metadata."
                        )

                        continue

                    # Decode base64 image
                    image_bytes = base64.b64decode(
                        image_base64
                    )

                    pil_image = Image.open(
                        io.BytesIO(image_bytes)
                    ).convert("RGB")

                    # Generate image embedding using CLIP
                    image_embedding = (
                        EmbeddingManager.embed_image(
                            pil_image
                        )
                    )

                    document.metadata["embedding_type"] = (
                        "image_embedding"
                    )

                    all_documents.append(document)
                    all_embeddings.append(image_embedding)

                # ===================================================
                # UNKNOWN MODALITY
                # ===================================================
                else:

                    logging.warning(
                        f"Unknown modality encountered: {modality}"
                    )

            # =======================================================
            # FINAL LOGGING
            # =======================================================
            logging.info(
                f"Generated {len(all_documents)} "
                f"multi-modal chunks successfully."
            )

            return all_documents, all_embeddings

        except Exception as e:
            logging.exception(
                "Error while processing multi-modal chunks."
            )
            raise CustomException(e, sys)