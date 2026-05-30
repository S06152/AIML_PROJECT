# Standard library imports
import io
import os
import sys
import tempfile
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
import fitz 
import pdfplumber
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFLoader:
    """
    Loads an AUTOSAR SWS PDF document uploaded via Streamlit into
    LangChain Document objects.

    Design:
        - Accepts a Streamlit UploadedFile object.
        - Validates that the file is a PDF.
        - Writes to a named temporary file (required by PyPDFLoader).
        - Extracts per-page Document objects.
        - Guarantees temp file cleanup via finally block.

    Usage:
        loader = PDFLoader(uploaded_file)
        documents = loader.load_documents()
    """

    def __init__(self, uploaded_file, user_input: dict) -> None:
        """
        Initialize PDFLoader with a Streamlit uploaded file.

        Args:
            uploaded_file: Streamlit UploadedFile object (file-like with .type attribute).

        Raises:
            Raises Exception: If file type is not PDF.
        """
        try:
            logging.info("Initializing PDFLoader.")
            
            if uploaded_file.type == "application/pdf":
                self._uploaded_file = uploaded_file
                logging.info("PDF file accepted")

            self._processor = BlipProcessor.from_pretrained(user_input['CAPTION_MODEL'])
            self._model = BlipForConditionalGeneration.from_pretrained(user_input['CAPTION_MODEL'])

            self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size    = user_input['CHUNK_SIZE'],
            chunk_overlap = user_input['CHUNK_OVERLAP'],
            separators    = ["\n\n", "\n", ".", " ", ""],
        )

        except Exception as e:
            logging.exception("Error during PDFLoader initialization.")
            raise CustomException(e, sys)

    def load_documents(self) -> List[Document]:
        """
        Extract per-page Document objects from the uploaded AUTOSAR SWS PDF.

        Workflow:
            1. Write uploaded bytes to a named temporary file.
            2. Pass the temp file path to PyPDFLoader.
            3. Extract documents (one per page).
            4. Delete the temp file in the finally block.
            5. Return extracted documents.

        Returns:
            List[Document]: One Document per PDF page with page_content + metadata.

        Raises:
            AutosarMASException: If reading, writing, or extraction fails.
        """

        temp_file_path : str = None

        try:
            logging.info("Saving uploaded PDF file to temporary location.")

            # Save uploaded file to temporary disk file
            # Required because PyPDFLoader expects a file path
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp:
                tmp.write(self._uploaded_file.read())
                temp_file_path = tmp.name

            logging.info(f"Temporary PDF file created at: {temp_file_path}")

            # Load PDF content into LangChain Documents
            text_docs  = self._extract_text(temp_file_path)
            table_docs = self._extract_tables(temp_file_path)
            image_docs = self._extract_images(temp_file_path)
            all_docs   = text_docs + table_docs + image_docs
            logging.info("Total extracted documents: %d", len(all_docs))
           
            return all_docs

        except Exception as e:
            logging.exception("Error while loading PDF documents.")
            raise CustomException(e, sys)

        finally:
            # Ensure temporary file cleanup even if an error occurs
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logging.info("Temporary PDF file deleted successfully.")
    
    def _extract_text(self, file_path: str) -> List[Document]:
        """
        Extract plain text using PyMuPDF and split into overlapping chunks.

        Returns
        -------
        list[Document]
            One Document per chunk, with source / page / content_type metadata.
        """
        docs: List[Document] = []

        try:
            with fitz.open(file_path) as pdf:
                for page_num, page in enumerate(pdf, start = 1):
                    try:
                        raw = page.get_text("text").strip()
                        if not raw:
                            continue
                        for chunk_idx, chunk in enumerate(self._text_splitter.split_text(raw)):
                            docs.append(
                                Document(
                                    page_content=chunk,
                                    metadata={
                                        "source"       : Path(file_path).name,
                                        "page"         : page_num,
                                        "chunk_index"  : chunk_idx,
                                        "content_type" : "text",
                                        "chunk_id"     : str(uuid.uuid4()),
                                    },
                                )
                            )
                    except Exception as exc:           
                        logging.warning("Text extraction failed page=%d: %s", page_num, exc)

        except Exception as exc:                    
            logging.error("Cannot open PDF for text extraction: %s", exc)

        logging.info("Text extraction: %d chunks", len(docs))
        
        return docs
    
    # Tables
    @staticmethod
    def _table_to_markdown(table: list[list[str | None]]) -> str:
        """
        Convert a pdfplumber table (list of rows) to a GitHub-Flavoured
        Markdown table string.

        Returns an empty string when the table is empty or malformed.
        """
        if not table or not table[0]:
            return ""

        def _cell(v: str | None) -> str:
            return str(v).replace("|", "\\|").strip() if v is not None else ""

        header = table[0]
        rows   = table[1:]
        lines  = [
            "| " + " | ".join(_cell(c) for c in header) + " |",
            "| " + " | ".join("---" for _ in header)   + " |",
        ]
        for row in rows:
            padded = list(row) + [None] * max(0, len(header) - len(row))
            lines.append("| " + " | ".join(_cell(c) for c in padded[: len(header)]) + " |")
        return "\n".join(lines)
    
    def _extract_tables(self, pdf_path: str) -> list[Document]:
        """
        Extract all tables from the PDF via pdfplumber and return them as
        Markdown Documents with source metadata.
        """
        docs: list[Document] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        tables = page.extract_tables()
                    except Exception as exc:           
                        logging.warning("Table extraction failed page=%d: %s", page_num, exc)
                        continue

                    for t_idx, table in enumerate(tables):
                        md = self._table_to_markdown(table)
                        if not md.strip():
                            continue
                        docs.append(
                            Document(
                                page_content=(
                                    f"[Table — page {page_num}, table {t_idx + 1}]\n{md}"
                                ),
                                metadata={
                                    "source"       : Path(pdf_path).name,
                                    "page"         : page_num,
                                    "table_index"  : t_idx,
                                    "content_type" : "table",
                                    "chunk_id"     : str(uuid.uuid4()),
                                },
                            )
                        )
        except Exception as exc:                       
            logging.error("Cannot open PDF for table extraction: %s", exc)

        logging.info("Table extraction: %d tables", len(docs))
        return docs
    
    # Images
    def _generate_caption(self, pil_image: Image.Image) -> str:
        """
        Run BLIP image captioning on a PIL image.

        Returns
        -------
        str
            Human-readable caption string.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)

        inputs = self._processor(
            images=pil_image, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=128
            )

        return self._processor.decode(
            output_ids[0], skip_special_tokens=True
        )

    def _extract_images(self, pdf_path: str) -> list[Document]:
        """
        Extract all embedded images from the PDF, generate BLIP captions,
        and return them as text Documents.  Images smaller than 50×50 pixels
        are silently ignored (likely decorative icons or watermarks).
        """
        docs: list[Document] = []
        try:
            with fitz.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf, start=1):
                    for img_idx, img_info in enumerate(page.get_images(full=True)):
                        xref = img_info[0]
                        try:
                            base_img  = pdf.extract_image(xref)
                            pil_image = Image.open(
                                io.BytesIO(base_img["image"])
                            ).convert("RGB")

                            if pil_image.width < 50 or pil_image.height < 50:
                                continue

                            caption = self._generate_caption(pil_image)
                            docs.append(
                                Document(
                                    page_content=(
                                        f"[Image — page {page_num}, image {img_idx + 1}]\n"
                                        f"Visual description: {caption}"
                                    ),
                                    metadata={
                                        "source"       : Path(pdf_path).name,
                                        "page"         : page_num,
                                        "image_index"  : img_idx,
                                        "content_type" : "image",
                                        "caption"      : caption,
                                        "chunk_id"     : str(uuid.uuid4()),
                                    },
                                )
                            )
                        except Exception as exc:       
                            logging.warning(
                                "Skipping image xref=%d page=%d: %s",
                                xref, page_num, exc,
                            )
        except Exception as exc:                      
            logging.error("Cannot open PDF for image extraction: %s", exc)

        logging.info("Image extraction: %d captions", len(docs))
        return docs