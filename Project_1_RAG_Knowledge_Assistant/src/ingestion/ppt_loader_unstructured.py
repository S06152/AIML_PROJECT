# Standard library imports
import os
import sys
import tempfile
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPowerPointLoader

class PPTLoader:
    """
    Wrapper class for loading PPTX (PowerPoint) files into LangChain Document objects.

    This class:
    - Accepts an uploaded PPTX file (e.g., from Streamlit).
    - Saves it temporarily to disk (since loader expects a file path).
    - Extracts slide content using UnstructuredPowerPointLoader.
    - Deletes the temporary file after processing.
    """

    def __init__(self, uploaded_file):
        """
        Initialize with uploaded file object.

        Args:
            uploaded_file: File-like object (e.g., Streamlit uploaded file).
        """
        try:
            logging.info("Initializing PPTLoader.")
            self.uploaded_file = uploaded_file

        except Exception as e:
            logging.exception("Error during PPTLoader initialization.")
            raise CustomException(e, sys)

    def load_documents(self) -> List[Document]:
        """
        Load PPTX content and convert it into LangChain Document objects.

        Workflow:
            1. Save uploaded PPTX to temporary file.
            2. Use UnstructuredPowerPointLoader to extract slide text.
            3. Delete temporary file.
            4. Return extracted documents.

        Returns:
            List[Document]: Extracted document objects from PPTX.
                            Typically each slide becomes one Document.
        """
        temp_file_path = None

        try:
            logging.info("Saving uploaded PPTX file to temporary location.")

            # Save uploaded file to temporary disk file
            # Required because UnstructuredPowerPointLoader expects a file path
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".pptx") as tmp:
                tmp.write(self.uploaded_file.read())
                temp_file_path = tmp.name

            logging.info(f"Temporary PPTX file created at: {temp_file_path}")

            # Load PPTX content into LangChain Documents
            logging.info("Loading PPTX file using UnstructuredPowerPointLoader.")
            loader = UnstructuredPowerPointLoader(temp_file_path)
            documents = loader.load()

            logging.info(
                f"PPTX loaded successfully. Generated {len(documents)} document(s)."
            )

            return documents

        except Exception as e:
            logging.exception("Error while loading PPTX documents.")
            raise CustomException(e, sys)

        finally:
            # Ensure temporary file cleanup even if an error occurs
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logging.info("Temporary PPTX file deleted successfully.")