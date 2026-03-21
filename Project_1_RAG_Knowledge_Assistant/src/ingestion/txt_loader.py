# Standard library imports
import os
import sys
import tempfile
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

class TxtLoader:
    """
    Wrapper class for loading TXT files into LangChain Document objects.

    This class:
    - Accepts an uploaded TXT file (e.g., from Streamlit).
    - Saves it temporarily to disk (since TextLoader requires a file path).
    - Extracts text content using LangChain's TextLoader.
    - Deletes the temporary file after processing.
    """

    def __init__(self, uploaded_file):
        """
        Initialize with uploaded file object.

        Args:
            uploaded_file: File-like object (e.g., Streamlit uploaded file).
        """
        try:
            logging.info("Initializing TxtLoader.")
            self.uploaded_file = uploaded_file

        except Exception as e:
            logging.exception("Error during TxtLoader initialization.")
            raise CustomException(e, sys)

    def load_documents(self) -> List[Document]:
        """
        Load TXT content and convert it into LangChain Document objects.

        Workflow:
            1. Save uploaded TXT file to a temporary file.
            2. Use TextLoader to extract text.
            3. Delete the temporary file.
            4. Return extracted documents.

        Returns:
            List[Document]: Extracted document objects from TXT file.
                            Typically the entire file becomes one Document.
        """
        temp_file_path = None

        try:
            logging.info("Saving uploaded TXT file to temporary location.")

            # Save uploaded file to temporary disk file
            # Required because TextLoader expects a file path
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".txt") as tmp:
                tmp.write(self.uploaded_file.read())
                temp_file_path = tmp.name

            logging.info(f"Temporary TXT file created at: {temp_file_path}")

            # Load TXT content into LangChain Documents
            logging.info("Loading TXT file using LangChain TextLoader.")
            loader = TextLoader(temp_file_path)
            documents = loader.load()

            logging.info(
                f"TXT loaded successfully. Generated {len(documents)} document(s)."
            )

            return documents

        except Exception as e:
            logging.exception("Error while loading TXT documents.")
            raise CustomException(e, sys)

        finally:
            # Ensure temporary file cleanup even if an error occurs
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logging.info("Temporary TXT file deleted successfully.")