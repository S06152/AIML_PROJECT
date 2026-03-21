# Standard library import
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkingStrategy:
    """
    Handles document chunking for the RAG pipeline.

    Since LLMs have token limits, large documents must be split into
    smaller overlapping chunks before embedding and storing in a vector database.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize chunking configuration.

        Args:
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.
                                 Overlap preserves context across chunk boundaries.
        """
        try:
            logging.info(
                f"Initializing ChunkingStrategy with "
                f"chunk_size = {chunk_size}, chunk_overlap = {chunk_overlap}"
            )

            # Store configuration
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

            # RecursiveCharacterTextSplitter splits text hierarchically using separators.
            # It attempts larger separators first (paragraphs), then smaller ones (words, characters).
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap,
                length_function = len,  # Determines how text length is measured
                separators = ["\n\n", "\n", ". ", " ", ""]
            )

            logging.info("RecursiveCharacterTextSplitter initialized successfully.")

        except Exception as e:
            logging.exception("Error during ChunkingStrategy initialization.")
            raise CustomException(e, sys)

    def split_documents_into_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Split input documents into smaller chunks.

        Each resulting chunk becomes a separate Document object
        that will later be embedded and stored in a vector database.

        Args:
            documents (List[Document]): Original documents.

        Returns:
            List[Document]: Chunked documents ready for embedding.
        """
        try:
            logging.info(f"Splitting {len(documents)} documents into chunks.")

            # Prevent processing empty input
            if not documents:
                logging.warning("Received empty document list for chunking.")
                return []

            # Perform chunking operation
            chunks = self.splitter.split_documents(documents)

            logging.info(
                f"Chunking completed successfully. "
                f"Generated {len(chunks)} chunks."
            )

            return chunks

        except Exception as e:
            logging.exception("Error while splitting documents into chunks.")
            raise CustomException(e, sys)