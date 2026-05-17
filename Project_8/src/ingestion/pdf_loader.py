# Standard Library Imports
import os
import sys
import tempfile
from typing import List

# Third-Party Imports
import fitz  # PyMuPDF
import pdfplumber

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# Custom Imports
from src.utils.logger import logging
from src.utils.exception import CustomException

import warnings

warnings.filterwarnings("ignore")


class PDFLoader:
    """
    Multi-Modal PDF Loader for RAG Systems.

    Features:
        - Text Extraction
        - Table Extraction
        - Image Extraction
        - Metadata Enrichment
        - Multi-Modal Document Creation

    Supported Content:
        - Research Papers
        - Financial Reports
        - Technical Documentation
        - PDFs containing charts, tables, and figures

    Output:
        Returns LangChain Document objects enriched with:
            - modality type
            - page number
            - source information
            - extracted content
    """

    def __init__(self, uploaded_file) -> None:
        """
        Initialize PDFLoader.

        Args:
            uploaded_file:
                Streamlit uploaded PDF file object.

        Raises:
            Exception:
                If uploaded file is invalid.
        """

        try:
            logging.info("Initializing PDFLoader.")

            if (
                uploaded_file is None
                or uploaded_file.type != "application/pdf"
            ):
                raise ValueError(
                    "Invalid file uploaded. Please upload a PDF file."
                )

            self._uploaded_file = uploaded_file

            logging.info(
                "PDF file validated successfully."
            )

        except Exception as e:
            logging.exception(
                "Error during PDFLoader initialization."
            )
            raise CustomException(e, sys)

    def load_documents(self) -> List[Document]:
        """
        Load and process PDF into multi-modal LangChain documents.

        Extraction Pipeline:
            1. Extract text using PyPDFLoader
            2. Extract tables using pdfplumber
            3. Extract images/charts using PyMuPDF

        Returns:
            List[Document]:
                Multi-modal document chunks.

        Raises:
            Exception:
                If PDF processing fails.
        """

        temp_file_path = None

        try:
            logging.info(
                "Saving uploaded PDF to temporary storage."
            )

            # ---------------------------------------------------
            # Save Uploaded File Temporarily
            # ---------------------------------------------------
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(
                    self._uploaded_file.read()
                )

                temp_file_path = tmp_file.name

            logging.info(
                f"Temporary PDF created at: {temp_file_path}"
            )

            multimodal_documents: List[Document] = []

            # ===================================================
            # 1. TEXT EXTRACTION
            # ===================================================
            logging.info(
                "Starting text extraction using PyPDFLoader."
            )

            loader = PyPDFLoader(temp_file_path)

            text_documents = loader.load()

            for doc in text_documents:

                doc.metadata["modality"] = "text"

                multimodal_documents.append(doc)

            logging.info(
                f"Extracted {len(text_documents)} text chunks."
            )

            # ===================================================
            # 2. TABLE EXTRACTION
            # ===================================================
            logging.info(
                "Starting table extraction using pdfplumber."
            )

            with pdfplumber.open(temp_file_path) as pdf:

                for page_number, page in enumerate(pdf.pages):

                    tables = page.extract_tables()

                    if tables:

                        for table_index, table in enumerate(tables):

                            table_text = "\n".join(
                                [
                                    " | ".join(
                                        [
                                            str(cell)
                                            if cell is not None
                                            else ""
                                            for cell in row
                                        ]
                                    )
                                    for row in table
                                ]
                            )

                            if table_text.strip():

                                multimodal_documents.append(
                                    Document(
                                        page_content=table_text,
                                        metadata={
                                            "modality": "table",
                                            "page": page_number + 1,
                                            "table_index": table_index,
                                            "source": (
                                                self._uploaded_file.name
                                            ),
                                        },
                                    )
                                )

            logging.info(
                "Table extraction completed successfully."
            )

            # ===================================================
            # 3. IMAGE / CHART EXTRACTION
            # ===================================================
            logging.info(
                "Starting image extraction using PyMuPDF."
            )

            pdf_document = fitz.open(temp_file_path)

            for page_number in range(len(pdf_document)):

                page = pdf_document[page_number]

                image_list = page.get_images(full=True)

                if image_list:

                    for image_index, _ in enumerate(image_list):

                        multimodal_documents.append(
                            Document(
                                page_content=(
                                    f"Extracted image from "
                                    f"page {page_number + 1}"
                                ),
                                metadata={
                                    "modality": "image",
                                    "page": page_number + 1,
                                    "image_index": image_index,
                                    "source": (
                                        self._uploaded_file.name
                                    ),
                                },
                            )
                        )

            pdf_document.close()

            logging.info(
                "Image extraction completed successfully."
            )

            # ===================================================
            # FINAL SUMMARY
            # ===================================================
            logging.info(
                f"Total multi-modal documents created: "
                f"{len(multimodal_documents)}"
            )

            return multimodal_documents

        except Exception as e:
            logging.exception(
                "Error occurred while processing PDF."
            )
            raise CustomException(e, sys)

        finally:
            # ---------------------------------------------------
            # Cleanup Temporary File
            # ---------------------------------------------------
            if (
                temp_file_path
                and os.path.exists(temp_file_path)
            ):
                os.remove(temp_file_path)

                logging.info(
                    "Temporary PDF file deleted successfully."
                )