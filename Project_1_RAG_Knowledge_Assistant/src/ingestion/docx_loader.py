# Standard library imports
import os
import sys
import tempfile
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader

class DocxLoader:
    """
    Wrapper class for loading DOCX files into LangChain Document objects.

    This class:
    - Accepts an uploaded DOCX file (e.g., from Streamlit).
    - Saves it temporarily to disk (since LangChain loaders expect file paths).
    - Loads content using Docx2txtLoader.
    - Cleans up the temporary file after processing.
    """

    def __init__(self, uploaded_file):
        """
        Initialize with uploaded file object.

        Args:
            uploaded_file: File-like object (e.g., Streamlit uploaded file).
        """
        try:
            logging.info("Initializing DocxLoader.")
            self.uploaded_file = uploaded_file

        except Exception as e:
            logging.exception("Error during DocxLoader initialization.")
            raise CustomException(e, sys)

    def load_documents(self) -> List[Document]:
        """
        Load DOCX content and convert it into LangChain Document objects.

        Steps:
            1. Save uploaded file to temporary location.
            2. Use Docx2txtLoader to extract text.
            3. Delete temporary file.
            4. Return extracted documents.

        Returns:
            List[Document]: Extracted document objects from DOCX file.
        """
        temp_file_path = None

        try:
            logging.info("Saving uploaded DOCX file to temporary location.")

            # Save uploaded file to temporary disk file
            # Required because Docx2txtLoader expects a file path
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".docx") as tmp:
                tmp.write(self.uploaded_file.read())
                temp_file_path = tmp.name

            logging.info(f"Temporary DOCX file created at: {temp_file_path}")

            # Load DOCX content into LangChain Documents
            logging.info("Loading DOCX file using Docx2txtLoader.")
            loader = Docx2txtLoader(temp_file_path)
            documents = loader.load()

            logging.info(
                f"DOCX loaded successfully. Generated {len(documents)} document(s)."
            )

            return documents

        except Exception as e:
            logging.exception("Error while loading DOCX documents.")
            raise CustomException(e, sys)

        finally:
            # Ensure temporary file cleanup even if error occurs
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logging.info("Temporary DOCX file deleted successfully.")