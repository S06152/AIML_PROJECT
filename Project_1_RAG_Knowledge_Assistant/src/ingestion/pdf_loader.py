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
    Wrapper class for loading PDF files into LangChain Document objects.

    This class:
    - Accepts an uploaded PDF file (e.g., from Streamlit).
    - Saves it temporarily to disk (since PyPDFLoader requires a file path).
    - Extracts text page-wise using PyPDFLoader.
    - Deletes the temporary file after processing.
    """

    def __init__(self, uploaded_file):
        """
        Initialize with uploaded file object.

        Args:
            uploaded_file: File-like object (e.g., Streamlit uploaded file).
        """
        try:
            logging.info("Initializing PDFLoader.")
            self.uploaded_file = uploaded_file

        except Exception as e:
            logging.exception("Error during PDFLoader initialization.")
            raise CustomException(e, sys)

    def load_documents(self) -> List[Document]:
        """
        Load PDF content and convert it into LangChain Document objects.

        Workflow:
            1. Save uploaded PDF to temporary file.
            2. Use PyPDFLoader to extract content (page-wise).
            3. Delete temporary file.
            4. Return extracted documents.

        Returns:
            List[Document]: Extracted document objects from PDF.
                            Each page typically becomes one Document.
        """
        temp_file_path = None

        try:
            logging.info("Saving uploaded PDF file to temporary location.")

            # Save uploaded file to temporary disk file
            # Required because PyPDFLoader expects a file path
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp:
                tmp.write(self.uploaded_file.read())
                temp_file_path = tmp.name

            logging.info(f"Temporary PDF file created at: {temp_file_path}")

            # Load PDF content into LangChain Documents
            logging.info("Loading PDF file using PyPDFLoader.")
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

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