# Standard library imports
import os
import sys
import tempfile
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredExcelLoader

class XlsxLoader:
    """
    Wrapper class for loading XLSX (Excel) files into LangChain Document objects.

    This class:
    - Accepts an uploaded Excel file (e.g., from Streamlit).
    - Saves it temporarily to disk (since UnstructuredExcelLoader requires a file path).
    - Extracts structured text content from Excel sheets.
    - Deletes the temporary file after processing.
    """

    def __init__(self, uploaded_file):
        """
        Initialize the loader with an uploaded file object.

        Args:
            uploaded_file: File-like object (e.g., Streamlit uploaded file).
        """
        try:
            logging.info("Initializing XlsxLoader.")
            self.uploaded_file = uploaded_file

        except Exception as e:
            logging.exception("Error during XlsxLoader initialization.")
            raise CustomException(e, sys)

    def load_documents(self) -> List[Document]:
        """
        Load Excel content and convert it into LangChain Document objects.

        Workflow:
            1. Save uploaded XLSX file to a temporary location.
            2. Use UnstructuredExcelLoader to extract sheet content.
            3. Delete the temporary file.
            4. Return extracted documents.

        Returns:
            List[Document]: Extracted document objects from Excel file.
                            Each sheet or section may become a separate Document.
        """

        temp_file_path = None

        try:
            logging.info("Saving uploaded XLSX file to temporary location.")

            # Save uploaded file to temporary disk file
            # Required because loader expects a file path
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".xlsx") as tmp:
                tmp.write(self.uploaded_file.read())
                temp_file_path = tmp.name

            logging.info(f"Temporary XLSX file created at: {temp_file_path}")

            # Load Excel content using LangChain loader
            logging.info("Loading XLSX file using UnstructuredExcelLoader.")
            loader = UnstructuredExcelLoader(temp_file_path)
            documents = loader.load()

            logging.info(
                f"XLSX loaded successfully. Generated {len(documents)} document(s)."
            )

            return documents

        except Exception as e:
            logging.exception("Error while loading XLSX documents.")
            raise CustomException(e, sys)

        finally:
            # Ensure temporary file cleanup even if an error occurs
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logging.info("Temporary XLSX file deleted successfully.")