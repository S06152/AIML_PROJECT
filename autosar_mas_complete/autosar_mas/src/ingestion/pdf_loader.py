"""
pdf_loader.py — Multi-modal AUTOSAR SWS PDF ingestion module.

Extracts three content types from uploaded PDFs:
    1. TEXT   — via PyPDFLoader (per-page Documents)
    2. TABLES — via pdfplumber (structured as pipe-delimited text)
    3. IMAGES — via PyMuPDF / fitz (stored as temp files, paths in metadata)

All extracted content is returned as a unified List[Document] ready for
the ChunkingStrategy to process.

Design:
    - Single Responsibility: PDFLoader only loads and extracts — does not chunk.
    - Resource cleanup: temp files are cleaned up in the finally block.
    - Graceful degradation: table/image extraction failures are logged
      but do not abort the entire pipeline.
"""

import os
import sys
import tempfile
from typing import List
import fitz          # PyMuPDF — image extraction
import pdfplumber    # table extraction
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from src.utils.logger import logger
from src.utils.exception import CustomException


class PDFLoader:
    """
    Loads an AUTOSAR SWS PDF uploaded via Streamlit into LangChain Documents.

    Extracts:
        - Text content    → Document(page_content=text, metadata={"type": "text"})
        - Table content   → Document(page_content=table_text, metadata={"type": "table"})
        - Image references → Document(page_content=path_ref, metadata={"type": "image"})

    Usage:
        loader = PDFLoader(uploaded_file)
        documents = loader.load_documents()
    """

    def __init__(self, uploaded_file) -> None:
        """
        Initialize PDFLoader with a Streamlit UploadedFile object.

        Args:
            uploaded_file: Streamlit UploadedFile (must be application/pdf).

        Raises:
            CustomException: If file is None or not a PDF.
        """
        try:
            logger.info("Initializing PDFLoader.")

            if uploaded_file is None:
                raise ValueError("uploaded_file must not be None.")
            if uploaded_file.type != "application/pdf":
                raise ValueError(
                    f"Invalid file type: '{uploaded_file.type}'. Only PDF is accepted."
                )

            self._uploaded_file = uploaded_file
            logger.info("PDFLoader initialized for file: '%s'.", uploaded_file.name)

        except Exception as e:
            raise CustomException(e, sys) from e

    def load_documents(self) -> List[Document]:
        """
        Extract text, tables, and image references from the uploaded PDF.

        Returns:
            List[Document]: Unified list of Document objects from all content types.

        Raises:
            CustomException: If a critical extraction step fails.
        """
        temp_file_path: str = ""
        image_dir: str = ""

        try:
            # ── Save uploaded file to disk for library access ─────────────────
            logger.info("Writing uploaded PDF to temporary disk file.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(self._uploaded_file.read())
                temp_file_path = tmp.name
            logger.info("Temporary PDF created at: %s", temp_file_path)

            documents: List[Document] = []

            # ── 1. TEXT EXTRACTION ────────────────────────────────────────────
            logger.info("Extracting text via PyPDFLoader.")
            try:
                loader = PyPDFLoader(temp_file_path)
                text_docs: List[Document] = loader.load()
                for doc in text_docs:
                    doc.metadata["type"] = "text"
                    documents.append(doc)
                logger.info("Text extraction complete: %d pages.", len(text_docs))
            except Exception as text_err:
                logger.warning("Text extraction failed (non-fatal): %s", str(text_err))

            # ── 2. TABLE EXTRACTION ───────────────────────────────────────────
            logger.info("Extracting tables via pdfplumber.")
            try:
                with pdfplumber.open(temp_file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables):
                            if not table:
                                continue
                            # Format table rows as pipe-delimited text
                            rows = [
                                " | ".join(
                                    str(cell) if cell else "" for cell in row
                                )
                                for row in table
                            ]
                            table_text = "\n".join(rows)
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
                logger.info("Table extraction complete.")
            except Exception as table_err:
                logger.warning("Table extraction failed (non-fatal): %s", str(table_err))

            # ── 3. IMAGE EXTRACTION ───────────────────────────────────────────
            logger.info("Extracting images via PyMuPDF.")
            try:
                image_dir = tempfile.mkdtemp(prefix="autosar_pdf_images_")
                pdf_doc = fitz.open(temp_file_path)

                for page_index in range(len(pdf_doc)):
                    page = pdf_doc[page_index]
                    for img_index, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        base_image = pdf_doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext   = base_image["ext"]

                        image_filename = f"page_{page_index}_img_{img_index}.{image_ext}"
                        image_path = os.path.join(image_dir, image_filename)

                        with open(image_path, "wb") as fp:
                            fp.write(image_bytes)

                        documents.append(
                            Document(
                                page_content=f"[AUTOSAR SWS Image] Stored at: {image_path}",
                                metadata={
                                    "type": "image",
                                    "page": page_index,
                                    "image_index": img_index,
                                    "image_format": image_ext,
                                    "image_path": image_path,
                                },
                            )
                        )
                logger.info("Image extraction complete.")
            except Exception as img_err:
                logger.warning("Image extraction failed (non-fatal): %s", str(img_err))

            logger.info(
                "PDF ingestion complete. Total documents: %d.", len(documents)
            )
            return documents

        except Exception as e:
            raise CustomException(e, sys) from e

        finally:
            # ── Cleanup temporary PDF file ────────────────────────────────────
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info("Temporary PDF file removed.")
