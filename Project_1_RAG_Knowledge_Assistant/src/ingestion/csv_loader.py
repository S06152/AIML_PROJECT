# Standard library imports
import os
import sys
import tempfile
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader

class CSVLoaderWrapper:
    """
    Wrapper class for loading CSV files into LangChain Document objects.

    This class:
    - Accepts an uploaded file (e.g., from Streamlit)
    - Saves it temporarily to disk
    - Uses LangChain's CSVLoader to parse it
    - Cleans up the temporary file after processing
    """

    def __init__(self, uploaded_file):
        """
        Initialize with uploaded file object.

        Args:
            uploaded_file: File-like object (e.g., Streamlit uploaded file).
        """
        try:
            logging.info("Initializing CSVLoaderWrapper.")
            self.uploaded_file = uploaded_file

        except Exception as e:
            logging.exception("Error during CSVLoaderWrapper initialization.")
            raise CustomException(e, sys)

    def load_documents(self) -> List[Document]:
        """
        Load CSV data and convert it into LangChain Document objects.

        Steps:
            1. Save uploaded file to temporary location.
            2. Use CSVLoader to parse file.
            3. Delete temporary file.
            4. Return parsed documents.

        Returns:
            List[Document]: Extracted document objects from CSV.
        """
        temp_file_path = None

        try:
            logging.info("Saving uploaded CSV file to temporary location.")

            # Save uploaded file to a temporary file on disk
            # Required because CSVLoader expects a file path
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".csv") as tmp:
                tmp.write(self.uploaded_file.read())
                temp_file_path = tmp.name

            logging.info(f"Temporary file created at: {temp_file_path}")

            # Load CSV content into LangChain Documents
            logging.info("Loading CSV file using LangChain CSVLoader.")
            loader = CSVLoader(temp_file_path)
            documents = loader.load()

            logging.info(f"CSV loaded successfully. Generated {len(documents)} documents.")

            return documents

        except Exception as e:
            logging.exception("Error while loading CSV documents.")
            raise CustomException(e, sys)

        finally:
            # Ensure temporary file cleanup even if an error occurs
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logging.info("Temporary CSV file deleted successfully.")