"""
chunk.py — Document chunking strategy for AUTOSAR SWS documents.

Splits LangChain Document objects into overlapping text chunks suitable
for vector embedding and similarity search.

AUTOSAR SWS Context:
    SWS documents contain numbered requirement clauses (e.g., [SWS_Com_00001]).
    A chunk size of 1200 chars typically captures 2–4 complete clauses,
    preserving the semantic unit that embeddings need to work well.
    Overlap of 200 chars prevents context loss at chunk boundaries.
"""

import sys
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import logger
from src.utils.exception import CustomException

# ---------------------------------------------------------------------------
# AUTOSAR SWS chunking defaults
# ---------------------------------------------------------------------------
_DEFAULT_CHUNK_SIZE: int    = 1200
_DEFAULT_CHUNK_OVERLAP: int = 200
_SEPARATORS: List[str]      = ["\n\n", "\n", ". ", " ", ""]


class ChunkingStrategy:
    """
    Splits LangChain Documents into overlapping chunks for the RAG pipeline.

    Uses RecursiveCharacterTextSplitter which splits hierarchically:
        paragraphs → sentences → words → characters

    Usage:
        chunker = ChunkingStrategy(chunk_size=1200, chunk_overlap=200)
        chunks  = chunker.split_documents_into_chunks(documents)
    """

    def __init__(
        self,
        chunk_size: int    = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """
        Initialize ChunkingStrategy with configurable size and overlap.

        Args:
            chunk_size    (int): Maximum characters per chunk. Default: 1200.
            chunk_overlap (int): Overlapping characters between chunks. Default: 200.

        Raises:
            CustomException: If splitter initialization fails.
        """
        try:
            logger.info(
                "Initializing ChunkingStrategy: chunk_size=%d, chunk_overlap=%d.",
                chunk_size, chunk_overlap,
            )

            self._chunk_size: int    = chunk_size
            self._chunk_overlap: int = chunk_overlap

            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                length_function=len,
                separators=_SEPARATORS,
            )

            logger.info("RecursiveCharacterTextSplitter initialized.")

        except Exception as e:
            raise CustomException(e, sys) from e

    def split_documents_into_chunks(
        self, documents: List[Document]
    ) -> List[Document]:
        """
        Split a list of Documents into overlapping chunks.

        Args:
            documents (List[Document]): Full-page documents from PDFLoader.

        Returns:
            List[Document]: Chunked documents ready for embedding.

        Raises:
            CustomException: If splitting fails.
        """
        try:
            if not documents:
                logger.warning("Empty document list received. Returning empty list.")
                return []

            logger.info("Splitting %d documents into chunks.", len(documents))
            chunks: List[Document] = self._splitter.split_documents(documents)

            logger.info(
                "Chunking complete. Generated %d chunks from %d documents.",
                len(chunks), len(documents),
            )
            return chunks

        except Exception as e:
            raise CustomException(e, sys) from e
