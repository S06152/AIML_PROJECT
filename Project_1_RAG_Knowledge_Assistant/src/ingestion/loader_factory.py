# Standard library imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.docx_loader import DocxLoader
from src.ingestion.ppt_loader import PPTLoader
from src.ingestion.csv_loader import CSVLoaderWrapper
from src.ingestion.txt_loader import TxtLoader
from src.ingestion.xlsx_loader import XlsxLoader

class LoaderFactory:
    """
    Factory class responsible for returning the correct document loader
    based on the uploaded file's MIME type.

    This enables:
    - Clean separation of file handling logic
    - Easy extensibility (add new file types in one place)
    - Centralized validation for supported file formats
    """

    # Mapping of MIME types to their corresponding loader classes
    MIME_LOADER_MAP = {
        "application/pdf": PDFLoader,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxLoader,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": PPTLoader,
        "text/csv": CSVLoaderWrapper,
        "text/plain": TxtLoader,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": XlsxLoader,
    }

    @staticmethod
    def get_loader(uploaded_file):
        """
        Return an instance of the appropriate loader based on file MIME type.

        Args:
            uploaded_file: File-like object (e.g., Streamlit uploaded file)
                           Must contain `.type` attribute for MIME detection.

        Returns:
            Instance of corresponding loader class.

        Raises:
            CustomException: If file type is unsupported or any unexpected error occurs.
        """
        try:
            logging.info("Determining appropriate loader for uploaded file.")

            # Extract MIME type from uploaded file
            mime_type = uploaded_file.type
            logging.info(f"Uploaded file MIME type detected: {mime_type}")

            # Retrieve matching loader class
            loader_class = LoaderFactory.MIME_LOADER_MAP.get(mime_type)

            # Validate support
            if loader_class is None:
                logging.error(f"Unsupported file type received: {mime_type}")
                raise ValueError(f"Unsupported file type: {mime_type}")

            logging.info(f"Loader selected: {loader_class.__name__}")

            # Return instantiated loader
            return loader_class(uploaded_file)

        except Exception as e:
            logging.exception("Error while selecting document loader.")
            raise CustomException(e, sys)