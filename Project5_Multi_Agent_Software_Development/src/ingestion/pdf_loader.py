# Standard library imports
import os
import sys
import tempfile
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:
    """
    Loads an AUTOSAR SWS PDF document uploaded via Streamlit into
    LangChain Document objects.

    Design:
        - Accepts a Streamlit UploadedFile object.
        - Validates that the file is a PDF.
        - Writes to a named temporary file (required by PyPDFLoader).
        - Extracts per-page Document objects.
        - Guarantees temp file cleanup via finally block.

    Usage:
        loader = PDFLoader(uploaded_file)
        documents = loader.load_documents()
    """

    def __init__(self, uploaded_file) -> None:
        """
        Initialize PDFLoader with a Streamlit uploaded file.

        Args:
            uploaded_file: Streamlit UploadedFile object (file-like with .type attribute).

        Raises:
            Raises Exception: If file type is not PDF.
        """
        try:
            logging.info("Initializing PDFLoader.")
            
            if uploaded_file.type == "application/pdf":
                self._uploaded_file = uploaded_file
                logging.info("PDF file accepted")

        except Exception as e:
            logging.exception("Error during PDFLoader initialization.")
            raise CustomException(e, sys)

    def load_documents(self) -> List[Document]:
        """
        Extract per-page Document objects from the uploaded AUTOSAR SWS PDF.

        Workflow:
            1. Write uploaded bytes to a named temporary file.
            2. Pass the temp file path to PyPDFLoader.
            3. Extract documents (one per page).
            4. Delete the temp file in the finally block.
            5. Return extracted documents.

        Returns:
            List[Document]: One Document per PDF page with page_content + metadata.

        Raises:
            AutosarMASException: If reading, writing, or extraction fails.
        """

        temp_file_path : str = None

        try:
            logging.info("Saving uploaded PDF file to temporary location.")

            # Save uploaded file to temporary disk file
            # Required because PyPDFLoader expects a file path
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp:
                tmp.write(self._uploaded_file.read())
                temp_file_path = tmp.name

            logging.info(f"Temporary PDF file created at: {temp_file_path}")

            # Load PDF content into LangChain Documents
            logging.info("Loading PDF file using PyPDFLoader.")
            loader = PyPDFLoader(temp_file_path)
            documents: List[Document] = loader.load()

            logging.info(
                f"PDF loaded successfully. Generated {len(documents)} document(s)."
            )

            return documents

        except Exception as e:
            logging.exception("Error while loading PDF documents.")
            raise CustomException(e, sys)

        finally:
            # Ensure temporary file cleanup even if an error occurs
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logging.info("Temporary PDF file deleted successfully.")