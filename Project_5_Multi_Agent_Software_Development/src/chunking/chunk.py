# Standard library import
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# AUTOSAR SWS document chunking defaults
# Larger chunks (1200 chars) preserve SWS requirement context across clauses.
# Overlap (200 chars) prevents context loss at chunk boundaries.
# ---------------------------------------------------------------------------
_DEFAULT_CHUNK_SIZE: int = 1200
_DEFAULT_CHUNK_OVERLAP: int = 200
_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]

class ChunkingStrategy:
    """
    Splits LangChain Document objects into overlapping chunks for the RAG pipeline.

    AUTOSAR SWS documents are structured with numbered requirement clauses.
    Chunk size of 1200 chars typically contains 2–4 SWS clauses, preserving
    semantic coherence for the embedding model.

    Usage:
        chunker = ChunkingStrategy(chunk_size=1200, chunk_overlap=200)
        chunks = chunker.split_documents_into_chunks(documents)
    """

    def __init__(self, chunk_size: int = _DEFAULT_CHUNK_SIZE, chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP)-> None:
        """
        Initialize the chunking strategy with size and overlap parameters.

        Args:
            chunk_size    (int): Max characters per chunk (default: 1200).
            chunk_overlap (int): Overlapping characters between chunks (default: 200).

        Raises:
            Raises Exception: If splitter initialization fails.
        """

        try:
            logging.info(
                f"Initializing ChunkingStrategy with "
                f"chunk_size = {chunk_size}, chunk_overlap = {chunk_overlap}"
            )

            # Store configuration
            self._chunk_size: int = chunk_size
            self._chunk_overlap: int = chunk_overlap

            # RecursiveCharacterTextSplitter splits text hierarchically using separators.
            # It attempts larger separators first (paragraphs), then smaller ones (words, characters).
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size = self._chunk_size,
                chunk_overlap = self._chunk_overlap,
                length_function = len,  # Determines how text length is measured
                separators = _SEPARATORS
            )

            logging.info("RecursiveCharacterTextSplitter initialized successfully.")

        except Exception as e:
            logging.exception("Error during ChunkingStrategy initialization.")
            raise CustomException(e, sys)

    def split_documents_into_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Document objects into overlapping chunks.

        Args:
            documents (List[Document]): Full-page documents from PDFLoader.

        Returns:
            List[Document]: Chunked documents ready for embedding.

        Raises:
            Raises Exception: If splitting fails.
        """
        try:
            logging.info(f"Splitting {len(documents)} documents into chunks.")

            # Prevent processing empty input
            if not documents:
                logging.warning("Received empty document list for chunking.")
                return []

            # Perform chunking operation
            chunks: List[Document] = self._splitter.split_documents(documents)

            logging.info(
                f"Chunking completed successfully. "
                f"Generated {len(chunks)} chunks."
            )

            return chunks

        except Exception as e:
            logging.exception("Error while splitting documents into chunks.")
            raise CustomException(e, sys)