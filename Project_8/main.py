"""
Multimodal RAG Pipeline — Streamlit App (Production / Interview Grade)
=======================================================================
Open-source stack (100 % free, no proprietary API required):

  PDF parsing     : PyMuPDF (fitz) + pdfplumber
  Image caption   : BLIP (Salesforce/blip-image-captioning-base)
  Embeddings      : sentence-transformers/all-MiniLM-L6-v2
  Vector store    : ChromaDB (EphemeralClient — in-memory per session)
  LLM             : Groq API — llama-3.3-70b-versatile
                    (FREE tier available at console.groq.com)

Secrets (Streamlit Cloud → Settings → Secrets):
    GROQ_API_KEY = "gsk_..."     # free Groq API key

Local run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import uuid
import tempfile
import logging
from pathlib import Path
from typing import Generator

import streamlit as st

# ── PDF ────────────────────────────────────────────────────────────────────
import fitz          # PyMuPDF
import pdfplumber
from PIL import Image

# ── ML ─────────────────────────────────────────────────────────────────────
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ── LangChain ──────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── ChromaDB ───────────────────────────────────────────────────────────────
import chromadb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Central configuration — single place to change models / knobs."""

    # ── Open-source models ────────────────────────────────────────────────
    EMBED_MODEL   : str = "sentence-transformers/all-MiniLM-L6-v2"
    CAPTION_MODEL : str = "Salesforce/blip-image-captioning-base"

    # Groq LLM — fast inference, free tier available at console.groq.com
    # Alternatives (swap freely):
    #   "llama-3.1-8b-instant"
    #   "mixtral-8x7b-32768"
    #   "gemma2-9b-it"
    LLM_MODEL_NAME  : str   = "llama-3.3-70b-versatile"
    LLM_MAX_TOKENS  : int   = 1024
    LLM_TEMPERATURE : float = 0.1

    # ── Chunking ──────────────────────────────────────────────────────────
    CHUNK_SIZE    : int = 800
    CHUNK_OVERLAP : int = 150

    # ── Retrieval ─────────────────────────────────────────────────────────
    TOP_K         : int = 5

    # ── ChromaDB shared collection ────────────────────────────────────────
    COLLECTION    : str = "multimodal_rag"

    # ── Hardware ──────────────────────────────────────────────────────────
    DEVICE        : str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def validate(cls) -> None:
        """Called once at startup; raises ValueError if critical values missing."""
        token = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        if not token:
            raise ValueError(
                "GROQ_API_KEY secret not set. "
                "Add it in Streamlit Cloud → Settings → Secrets, or export GROQ_API_KEY locally."
            )


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 1 — PDF EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

class PDFExtractor:
    """
    Extracts three content types from a PDF file:

    - Text   → PyMuPDF page text → chunked with RecursiveCharacterTextSplitter
    - Tables → pdfplumber       → Markdown strings
    - Images → PyMuPDF          → BLIP captions → text Documents

    Parameters
    ----------
    caption_processor : BlipProcessor
        Pre-loaded BLIP tokeniser/processor (shared, never re-created).
    caption_model : BlipForConditionalGeneration
        Pre-loaded BLIP model (shared, never re-created).
    """

    def __init__(
        self,
        caption_processor: BlipProcessor,
        caption_model: BlipForConditionalGeneration,
    ) -> None:
        self.caption_processor = caption_processor
        self.caption_model     = caption_model
        self.device            = Config.DEVICE

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size    = Config.CHUNK_SIZE,
            chunk_overlap = Config.CHUNK_OVERLAP,
            separators    = ["\n\n", "\n", ".", " ", ""],
        )

    # ── Text ───────────────────────────────────────────────────────────────

    def _extract_text(self, pdf_path: str) -> list[Document]:
        """
        Extract plain text using PyMuPDF and split into overlapping chunks.

        Returns
        -------
        list[Document]
            One Document per chunk, with source / page / content_type metadata.
        """
        docs: list[Document] = []
        try:
            with fitz.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf, start=1):
                    try:
                        raw = page.get_text("text").strip()
                        if not raw:
                            continue
                        for chunk_idx, chunk in enumerate(
                            self._text_splitter.split_text(raw)
                        ):
                            docs.append(
                                Document(
                                    page_content=chunk,
                                    metadata={
                                        "source"       : Path(pdf_path).name,
                                        "page"         : page_num,
                                        "chunk_index"  : chunk_idx,
                                        "content_type" : "text",
                                        "chunk_id"     : str(uuid.uuid4()),
                                    },
                                )
                            )
                    except Exception as exc:           # noqa: BLE001
                        log.warning("Text extraction failed page=%d: %s", page_num, exc)
        except Exception as exc:                       # noqa: BLE001
            log.error("Cannot open PDF for text extraction: %s", exc)

        log.info("Text extraction: %d chunks", len(docs))
        return docs

    # ── Tables ─────────────────────────────────────────────────────────────

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
                    except Exception as exc:           # noqa: BLE001
                        log.warning("Table extraction failed page=%d: %s", page_num, exc)
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
        except Exception as exc:                       # noqa: BLE001
            log.error("Cannot open PDF for table extraction: %s", exc)

        log.info("Table extraction: %d tables", len(docs))
        return docs

    # ── Images ─────────────────────────────────────────────────────────────

    def _generate_caption(self, pil_image: Image.Image) -> str:
        """
        Run BLIP image captioning on a PIL image.

        Returns
        -------
        str
            Human-readable caption string.
        """
        inputs = self.caption_processor(
            images=pil_image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.caption_model.generate(
                **inputs, max_new_tokens=128
            )

        return self.caption_processor.decode(
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
                        except Exception as exc:       # noqa: BLE001
                            log.warning(
                                "Skipping image xref=%d page=%d: %s",
                                xref, page_num, exc,
                            )
        except Exception as exc:                       # noqa: BLE001
            log.error("Cannot open PDF for image extraction: %s", exc)

        log.info("Image extraction: %d captions", len(docs))
        return docs

    def extract(self, pdf_path: str) -> list[Document]:
        """
        Run all three extractors and return a unified document list.

        Parameters
        ----------
        pdf_path : str
            Absolute path to the PDF file on disk.

        Returns
        -------
        list[Document]
            Combined list of text chunks, table rows, and image captions.
        """
        text_docs  = self._extract_text(pdf_path)
        table_docs = self._extract_tables(pdf_path)
        image_docs = self._extract_images(pdf_path)
        all_docs   = text_docs + table_docs + image_docs
        log.info("Total extracted documents: %d", len(all_docs))
        return all_docs


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 2 — EMBEDDING MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingManager:
    """
    Wraps sentence-transformers embeddings via langchain-huggingface.

    The underlying model is loaded once and reused across all indexing calls.
    """

    def __init__(self, model_name: str = Config.EMBED_MODEL) -> None:
        log.info(
            "Loading embedding model: %s (device=%s)", model_name, Config.DEVICE
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name    = model_name,
            model_kwargs  = {"device": Config.DEVICE},
            encode_kwargs = {"normalize_embeddings": True},
        )

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Return the LangChain-compatible embedding object."""
        return self._embeddings


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 3 — VECTOR STORE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class VectorStoreManager:
    """
    Manages a single in-memory ChromaDB collection that accumulates documents
    from ALL indexed PDFs within a session.
    """

    def __init__(self, embedding_manager: EmbeddingManager) -> None:
        self._embedding_manager = embedding_manager
        self._chroma_client     = chromadb.EphemeralClient()
        self._vectorstore: Chroma | None = None

    def add_documents(self, documents: list[Document]) -> None:
        """
        Embed and add documents to the shared ChromaDB collection.

        Safe to call multiple times — subsequent calls append to the existing
        collection rather than overwriting it.
        """
        log.info(
            "Adding %d documents to ChromaDB collection '%s'",
            len(documents), Config.COLLECTION,
        )
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                client             = self._chroma_client,
                collection_name    = Config.COLLECTION,
                embedding_function = self._embedding_manager.get_embeddings(),
            )

        self._vectorstore.add_documents(documents)
        log.info("ChromaDB updated — total searchable across all PDFs.")

    def get_retriever(self, k: int = Config.TOP_K):
        """
        Return a LangChain retriever over the full collection.

        Raises
        ------
        RuntimeError
            If no documents have been indexed yet.
        """
        if self._vectorstore is None:
            raise RuntimeError("No documents indexed yet — call add_documents() first.")
        return self._vectorstore.as_retriever(
            search_type   = "similarity",
            search_kwargs = {"k": k},
        )

    def is_ready(self) -> bool:
        """Return True once at least one batch of documents has been indexed."""
        return self._vectorstore is not None


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 4 — GROQ LLM MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class OpenSourceLLMManager:
    """
    Wraps the Groq Inference API via LangChain's ChatGroq integration.

    Model: llama-3.3-70b-versatile (free tier available on groq.com).
    Requires GROQ_API_KEY set in Streamlit secrets or as an environment variable.

    The prompt chain uses system + human turns via ChatPromptTemplate → StrOutputParser.
    """

    _SYSTEM = (
        "You are an expert document analyst. "
        "Answer questions strictly from the provided context, which may contain "
        "text excerpts, Markdown tables, and image captions extracted from one "
        "or more PDF documents.\n\n"
        "Rules:\n"
        "- Answer concisely and accurately using ONLY the context.\n"
        "- If information spans multiple PDFs, state which document each fact "
        "comes from.\n"
        "- For table data, reference the relevant rows and columns explicitly.\n"
        "- For image data, reference the visual description.\n"
        "- If the context is insufficient, say: "
        "'I don't have enough information to answer this question.'"
    )

    _HUMAN = "Context:\n{context}\n\nQuestion: {question}"

    def __init__(self, groq_api_key: str) -> None:
        log.info("Initialising Groq LLM: %s", Config.LLM_MODEL_NAME)

        llm = ChatGroq(
            model       = Config.LLM_MODEL_NAME,
            temperature = Config.LLM_TEMPERATURE,
            max_tokens  = Config.LLM_MAX_TOKENS,
            api_key     = groq_api_key,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._SYSTEM),
            ("human",  self._HUMAN),
        ])

        # Chain: prompt → LLM → plain string
        self._chain = prompt | llm | StrOutputParser()

    def generate(self, context: str, question: str) -> str:
        """
        Generate an answer grounded in the retrieved context.

        Parameters
        ----------
        context  : str  Concatenated retrieved document chunks.
        question : str  User's natural-language question.

        Returns
        -------
        str  Stripped answer string.
        """
        result = self._chain.invoke({"context": context, "question": question})
        return result.strip() if isinstance(result, str) else str(result).strip()

    def stream(self, context: str, question: str) -> Generator[str, None, None]:
        """
        Stream the answer token-by-token (used for live Streamlit rendering).
        """
        yield from self._chain.stream({"context": context, "question": question})


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 5 — RAG PIPELINE (Orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    End-to-end multi-PDF orchestrator.

    Usage
    -----
    pipeline = RAGPipeline()
    pipeline.index_one(uploaded_file, status_callback=fn)   # repeat per PDF
    result   = pipeline.query("What is the revenue in Q3?")
    """

    def __init__(self) -> None:
        # Retrieve Groq API key once at construction time
        self._groq_api_key = st.secrets.get(
            "GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")
        )

        self._embedding_manager    = _load_embedding_manager()
        self._vector_store_manager = VectorStoreManager(self._embedding_manager)
        self._llm_manager          = OpenSourceLLMManager(
            groq_api_key=self._groq_api_key
        )

        # Registry: filename → document count
        self.indexed_pdfs: dict[str, int] = {}

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _build_context(docs: list[Document]) -> str:
        """Serialise retrieved documents into a numbered context block."""
        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            ctype  = doc.metadata.get("content_type", "text").upper()
            page   = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "unknown")
            parts.append(
                f"[{i}] ({ctype} | {source} | Page {page})\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Indexing ───────────────────────────────────────────────────────────

    def index_one(
        self,
        uploaded_file,
        status_callback=None,
    ) -> int:
        """
        Parse a single PDF and add its documents to the shared index.

        Parameters
        ----------
        uploaded_file : UploadedFile
            Streamlit uploaded file object (has .name and .read() ).
        status_callback : callable, optional
            Receives progress strings as they are emitted.

        Returns
        -------
        int
            Number of documents added (0 if already indexed).
        """
        pdf_name = uploaded_file.name

        if pdf_name in self.indexed_pdfs:
            msg = f"'{pdf_name}' already indexed — skipping."
            log.warning(msg)
            if status_callback:
                status_callback(f"ℹ️ {msg}")
            return 0

        def _emit(msg: str) -> None:
            log.info(msg)
            if status_callback:
                status_callback(msg)

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            cap_processor, cap_model = _load_caption_model()
            extractor = PDFExtractor(
                caption_processor=cap_processor,
                caption_model=cap_model,
            )

            _emit(f"📄 [{pdf_name}] Extracting text…")
            text_docs = extractor._extract_text(tmp_path)
            _emit(f"   ✅ {len(text_docs)} text chunks")

            _emit(f"📊 [{pdf_name}] Extracting tables…")
            table_docs = extractor._extract_tables(tmp_path)
            _emit(f"   ✅ {len(table_docs)} tables")

            _emit(f"🖼️ [{pdf_name}] Captioning images…")
            if Config.DEVICE == "cpu":
                _emit("   ⏳ CPU mode — image captioning may take a moment…")
            image_docs = extractor._extract_images(tmp_path)
            _emit(f"   ✅ {len(image_docs)} image captions")

            new_docs = text_docs + table_docs + image_docs
            _emit(
                f"📥 [{pdf_name}] Embedding {len(new_docs)} documents into ChromaDB…"
            )
            self._vector_store_manager.add_documents(new_docs)
            self.indexed_pdfs[pdf_name] = len(new_docs)
            _emit(f"🎉 [{pdf_name}] Done!")

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError as exc:
                    log.warning("Could not delete temp file %s: %s", tmp_path, exc)

        return self.indexed_pdfs.get(pdf_name, 0)

    # ── Querying ───────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Retrieve relevant chunks and generate an answer across all indexed PDFs.

        Parameters
        ----------
        question : str  User's natural-language question.

        Returns
        -------
        dict  Keys: answer (str), sources (list[dict]), top_docs (list[Document]).
        """
        if not self._vector_store_manager.is_ready():
            raise RuntimeError("No PDFs indexed. Call index_one() first.")

        retriever = self._vector_store_manager.get_retriever(k=Config.TOP_K)
        top_docs  = retriever.invoke(question)
        log.info("Retrieved %d documents for question: %s", len(top_docs), question)

        context = self._build_context(top_docs)
        answer  = self._llm_manager.generate(context=context, question=question)

        sources = [
            {
                "content_type": d.metadata.get("content_type", "text"),
                "page"        : d.metadata.get("page", "?"),
                "source"      : d.metadata.get("source", "unknown"),
            }
            for d in top_docs
        ]

        return {"answer": answer, "sources": sources, "top_docs": top_docs}

    def stream_query(self, question: str):
        """
        Like query() but yields answer tokens one-by-one for live rendering.

        Yields
        ------
        str  Token fragments from the LLM stream.
        """
        if not self._vector_store_manager.is_ready():
            raise RuntimeError("No PDFs indexed. Call index_one() first.")

        retriever = self._vector_store_manager.get_retriever(k=Config.TOP_K)
        top_docs  = retriever.invoke(question)
        context   = self._build_context(top_docs)

        sources = [
            {
                "content_type": d.metadata.get("content_type", "text"),
                "page"        : d.metadata.get("page", "?"),
                "source"      : d.metadata.get("source", "unknown"),
            }
            for d in top_docs
        ]

        yield from self._llm_manager.stream(context=context, question=question)
        return sources

    @property
    def is_ready(self) -> bool:
        """True once at least one PDF has been fully indexed."""
        return self._vector_store_manager.is_ready()

    @property
    def pdf_count(self) -> int:
        return len(self.indexed_pdfs)

    @property
    def pdf_names(self) -> list[str]:
        return list(self.indexed_pdfs.keys())


# ═══════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADERS  (Streamlit resource cache — loaded once per process)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ Loading sentence-transformer embeddings…")
def _load_embedding_manager() -> EmbeddingManager:
    """Cached embedding manager — singleton per Streamlit process."""
    return EmbeddingManager()


@st.cache_resource(show_spinner="⏳ Loading BLIP image captioning model…")
def _load_caption_model() -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    """
    Cached BLIP model — singleton per Streamlit process.

    Using float16 on CUDA halves VRAM usage; float32 is required on CPU.
    """
    dtype     = torch.float16 if Config.DEVICE == "cuda" else torch.float32
    processor = BlipProcessor.from_pretrained(Config.CAPTION_MODEL)
    model     = BlipForConditionalGeneration.from_pretrained(
        Config.CAPTION_MODEL, torch_dtype=dtype
    ).to(Config.DEVICE)
    model.eval()
    return processor, model


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 6 — STREAMLIT APP (UI Layer)
# ═══════════════════════════════════════════════════════════════════════════

class StreamlitApp:
    """
    Streamlit UI — dark editorial theme with animated status banners,
    colour-coded source badges, and a scrollable chat column.
    """

    # Emoji badges per content type
    _BADGE: dict[str, str] = {
        "text"  : "📝",
        "table" : "📊",
        "image" : "🖼️",
    }

    # Inline colour per content type (used in source pills)
    _PILL_COLOR: dict[str, str] = {
        "text"  : "#3b82f6",   # blue
        "table" : "#10b981",   # emerald
        "image" : "#f59e0b",   # amber
    }

    def __init__(self) -> None:
        st.set_page_config(
            page_title = "Multimodal PDF RAG",
            page_icon  = "📚",
            layout     = "wide",
        )
        self._inject_css()
        self._init_session_state()

    # ── CSS ────────────────────────────────────────────────────────────────

    @staticmethod
    def _inject_css() -> None:
        """Inject a dark editorial theme with custom source-pill styles."""
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

            html, body, [class*="css"] {
                font-family: 'IBM Plex Sans', sans-serif;
                background-color: #0f1117;
                color: #e2e8f0;
            }

            [data-testid="stSidebar"] {
                background: #1a1d27;
                border-right: 1px solid #2d3148;
            }

            .src-pill {
                display: inline-block;
                padding: 2px 10px;
                border-radius: 999px;
                font-size: 0.75rem;
                font-family: 'IBM Plex Mono', monospace;
                margin: 2px 4px 2px 0;
                font-weight: 600;
                color: #fff;
            }

            .stChatMessage {
                border-radius: 12px;
                margin-bottom: 8px;
            }

            h1, h2, h3 {
                font-family: 'IBM Plex Sans', sans-serif;
                font-weight: 600;
                letter-spacing: -0.02em;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # ── Session state ──────────────────────────────────────────────────────

    @staticmethod
    def _init_session_state() -> None:
        defaults: dict = {
            "pipeline"     : None,
            "chat_history" : [],
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # ── Sidebar ────────────────────────────────────────────────────────────

    def _render_sidebar(self) -> tuple[list, bool]:
        """Render sidebar and return (uploaded_files, index_clicked)."""
        with st.sidebar:
            st.markdown("## 📚 Multimodal PDF RAG")
            st.caption("Text · Tables · Images  |  Groq LLaMA 3.3 70B")
            st.divider()

            token_set = bool(
                st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
            )
            if token_set:
                st.success("🔑 GROQ_API_KEY detected", icon="✅")
            else:
                st.error(
                    "🔑 GROQ_API_KEY not set. "
                    "Add it under Settings → Secrets.",
                    icon="❌",
                )

            st.divider()

            uploaded_files = st.file_uploader(
                "📄 Upload PDF(s)",
                type                  = ["pdf"],
                accept_multiple_files = True,
                help                  = "Select one or more PDFs. "
                                        "Already-indexed files are skipped.",
            )

            index_clicked = st.button(
                "⚡ Index PDF(s)",
                disabled            = (not uploaded_files or not token_set),
                use_container_width = True,
                type                = "primary",
            )

            pipeline: RAGPipeline | None = st.session_state.pipeline
            if pipeline and pipeline.pdf_count > 0:
                st.divider()
                st.markdown(f"**📂 Indexed PDFs ({pipeline.pdf_count})**")
                for name, count in pipeline.indexed_pdfs.items():
                    st.markdown(f"- `{name}` &nbsp; ({count} docs)")

            if pipeline and pipeline.pdf_count > 0:
                st.divider()
                if st.button("🗑️ Clear all & restart", use_container_width=True):
                    st.session_state.pipeline     = None
                    st.session_state.chat_history = []
                    st.rerun()

            st.divider()
            st.markdown(
                """
**Open-source stack**
- 🤗 Embeddings: `all-MiniLM-L6-v2`
- 🖼️ Captions: `BLIP`
- 🗄️ Vector DB: `ChromaDB` (in-memory)
- 🚀 LLM: `llama-3.3-70b-versatile` (Groq)
- 🔎 Retrieval: similarity search
                """
            )

        return uploaded_files or [], index_clicked

    # ── Indexing ───────────────────────────────────────────────────────────

    def _handle_indexing(self, uploaded_files: list) -> None:
        """Create or reuse the pipeline and index every uploaded file."""
        if st.session_state.pipeline is None:
            st.session_state.pipeline = RAGPipeline()

        pipeline: RAGPipeline = st.session_state.pipeline

        for uploaded_file in uploaded_files:
            pdf_name = uploaded_file.name

            if pdf_name in pipeline.indexed_pdfs:
                st.info(f"ℹ️ **{pdf_name}** already indexed — skipped.")
                continue

            messages: list[str] = []

            def _cb(msg: str, _msgs: list = messages) -> None:
                _msgs.append(msg)

            with st.status(
                f"Processing **{pdf_name}**…", expanded=True
            ) as status_box:
                total = pipeline.index_one(uploaded_file, status_callback=_cb)
                for msg in messages:
                    status_box.write(msg)
                if total > 0:
                    status_box.update(
                        label = f"✅ Indexed **{total}** documents from `{pdf_name}`",
                        state = "complete",
                    )
                else:
                    status_box.update(
                        label = f"⏭️ Skipped `{pdf_name}` (already indexed)",
                        state = "complete",
                    )

        names = ", ".join(f"`{n}`" for n in pipeline.pdf_names)
        st.success(f"✅ Knowledge base active: {names}")

    # ── Source rendering ───────────────────────────────────────────────────

    def _render_sources(self, sources: list[dict]) -> None:
        """Render colour-coded source pills inside an expander."""
        if not sources:
            return
        with st.expander("📎 Retrieved sources", expanded=False):
            for s in sources:
                ctype  = s.get("content_type", "text")
                badge  = self._BADGE.get(ctype, "📄")
                color  = self._PILL_COLOR.get(ctype, "#6b7280")
                source = s.get("source", "unknown")
                page   = s.get("page", "?")
                st.markdown(
                    f'<span class="src-pill" style="background:{color};">'
                    f"{badge} {ctype.upper()}</span> "
                    f"`{source}` · Page {page}",
                    unsafe_allow_html=True,
                )

    # ── Chat history ───────────────────────────────────────────────────────

    def _render_chat_history(self) -> None:
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])
                self._render_sources(turn.get("sources", []))

    # ── Query ──────────────────────────────────────────────────────────────

    def _handle_query(self, question: str) -> None:
        """Retrieve, generate, and display the answer."""
        pipeline: RAGPipeline = st.session_state.pipeline

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()

            with st.spinner("🔎 Searching all PDFs and generating answer…"):
                result = pipeline.query(question)

            answer_placeholder.write(result["answer"])
            self._render_sources(result["sources"])

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer"  : result["answer"],
                "sources" : result["sources"],
            }
        )

    # ── Main ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Application entry point — called by Streamlit on every rerun."""

        try:
            Config.validate()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        uploaded_files, index_clicked = self._render_sidebar()

        if index_clicked and uploaded_files:
            self._handle_indexing(uploaded_files)

        pipeline: RAGPipeline | None = st.session_state.pipeline

        if pipeline is None or not pipeline.is_ready:
            st.markdown("## 📚 Multimodal PDF RAG")
            st.info(
                "👈 Upload one or more PDFs in the sidebar and click "
                "**Index PDF(s)** to begin."
            )
            st.markdown(
                """
                **What this system can answer:**
                - Questions about text content from your PDFs
                - Data extracted from tables
                - Visual descriptions of embedded images
                - Cross-document queries when multiple PDFs are loaded
                """
            )
            return

        if pipeline.pdf_count == 1:
            st.header(f"💬 Chatting with `{pipeline.pdf_names[0]}`")
        else:
            st.header(f"💬 Chatting across {pipeline.pdf_count} PDFs")
            cols = st.columns(min(pipeline.pdf_count, 4))
            for col, name in zip(cols, pipeline.pdf_names):
                col.markdown(f"📄 `{name}`")

        self._render_chat_history()

        question = st.chat_input(
            "Ask anything — answers draw from all indexed PDFs…"
        )
        if question:
            self._handle_query(question)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    StreamlitApp().run()
