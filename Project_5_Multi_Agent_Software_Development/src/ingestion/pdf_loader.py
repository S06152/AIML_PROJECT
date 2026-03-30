# Standard library imports
import os
import sys
import tempfile
from typing import List
import fitz         # PyMuPDF (for images)
import pdfplumber   # for tables
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from src.utils.logger import logging
from src.utils.exception import CustomException

class PDFLoader:
    """
    Loads an AUTOSAR SWS PDF document uploaded via Streamlit into
    LangChain Document objects.

    Enhanced Features:
        - Extracts TEXT (PyPDFLoader)
        - Extracts TABLES (pdfplumber)
        - Extracts IMAGES (PyMuPDF)
    """

    def __init__(self, uploaded_file) -> None:
        try:
            logging.info("Initializing PDFLoader.")

            # ✅ FIX: Proper validation
            if uploaded_file is None or uploaded_file.type != "application/pdf":
                raise ValueError("Invalid file. Please upload a PDF.")

            self._uploaded_file = uploaded_file
            logging.info("PDF file accepted")

        except Exception as e:
            logging.exception("Error during PDFLoader initialization.")
            raise CustomException(e, sys)

    def load_documents(self) -> List[Document]:

        temp_file_path: str = None

        try:
            logging.info("Saving uploaded PDF file to temporary location.")

            # Save uploaded file to temporary disk file
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp:
                tmp.write(self._uploaded_file.read())
                temp_file_path = tmp.name

            logging.info(f"Temporary PDF file created at: {temp_file_path}")

            documents: List[Document] = []

            # =====================================
            # 1. TEXT EXTRACTION (existing logic)
            # =====================================
            logging.info("Loading PDF file using PyPDFLoader.")
            loader = PyPDFLoader(temp_file_path)
            text_docs: List[Document] = loader.load()

            for doc in text_docs:
                doc.metadata["type"] = "text"
                documents.append(doc)

            logging.info(f"Extracted {len(text_docs)} text documents.")

            # =====================================
            # 2. TABLE EXTRACTION (NEW)
            # =====================================
            logging.info("Extracting tables from PDF.")

            with pdfplumber.open(temp_file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):

                    tables = page.extract_tables()

                    for table_idx, table in enumerate(tables):
                        table_text = "\n".join(
                            [
                                " | ".join(
                                    [str(cell) if cell else "" for cell in row]
                                )
                                for row in table
                            ]
                        )

                        documents.append(
                            Document(
                                page_content=table_text,
                                metadata={
                                    "type": "table",
                                    "page": page_num,
                                    "table_index": table_idx,
                                },
                            )
                        )

            # =====================================
            # 3. IMAGE EXTRACTION (NEW)
            # =====================================
            logging.info("Extracting images from PDF.")

            pdf_doc = fitz.open(temp_file_path)

            for page_index in range(len(pdf_doc)):
                page = pdf_doc[page_index]
                images = page.get_images(full=True)

                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = pdf_doc.extract_image(xref)

                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    documents.append(
                        Document(
                            page_content="Extracted image (binary data)",
                            metadata={
                                "type": "image",
                                "page": page_index,
                                "image_index": img_index,
                                "image_format": image_ext,
                                "image_bytes": image_bytes,
                            },
                        )
                    )

            logging.info(
                f"PDF processed successfully. Total documents created: {len(documents)}"
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