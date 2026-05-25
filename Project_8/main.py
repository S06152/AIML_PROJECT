"""
Multimodal RAG Pipeline — Streamlit Cloud App (Class-Based Architecture)
=========================================================================
Open-source stack:
  - PDF parsing   : PyMuPDF (fitz) + pdfplumber
  - Image caption : BLIP  (Salesforce/blip-image-captioning-base)
  - Embeddings    : HuggingFace all-MiniLM-L6-v2 via langchain-huggingface
  - Vector store  : ChromaDB  (EphemeralClient — in-memory per session)
  - LLM           : Groq  (llama-3.3-70b-versatile — free tier)
  - Framework     : LangChain + Streamlit

Secrets (Streamlit Cloud → Settings → Secrets):
    GROQ_API_KEY = "gsk_..."

Local run:
    pip install -r requirements.txt
    streamlit run app.py

Architecture (classes):
    PDFExtractor          — text / table / image extraction from PDF bytes
    EmbeddingManager      — HuggingFace embedding wrapper
    VectorStoreManager    — ChromaDB in-memory store (multi-PDF, single collection)
    GroqLLMManager        — Groq ChatGroq LLM wrapper
    RAGPipeline           — orchestrator: index + query (supports multiple PDFs)
    StreamlitApp          — UI layer
"""

from __future__ import annotations

import io
import os
import uuid
import tempfile
import logging
from pathlib import Path
from typing import Any

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
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ── ChromaDB ───────────────────────────────────────────────────────────────
import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Central configuration — change model names / knobs here."""

    # Models
    EMBED_MODEL   : str = "sentence-transformers/all-MiniLM-L6-v2"
    CAPTION_MODEL : str = "Salesforce/blip-image-captioning-base"
    GROQ_MODEL    : str = "llama-3.3-70b-versatile"

    # Chunking
    CHUNK_SIZE    : int = 800
    CHUNK_OVERLAP : int = 150

    # Retrieval
    TOP_K         : int = 5    # docs passed to LLM

    # ChromaDB shared collection name (all PDFs go into one collection)
    COLLECTION    : str = "multimodal_rag_multi"

    # Hardware
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 1 — PDF EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

class PDFExtractor:
    """
    Extracts three content types from a PDF file:
      • Text  → chunked via RecursiveCharacterTextSplitter
      • Tables→ pdfplumber → Markdown strings
      • Images→ PyMuPDF   → BLIP captions → text documents
    """

    def __init__(self, caption_processor: BlipProcessor, caption_model: BlipForConditionalGeneration):
        self.caption_processor = caption_processor
        self.caption_model     = caption_model
        self.device            = Config.DEVICE

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size    = Config.CHUNK_SIZE,
            chunk_overlap = Config.CHUNK_OVERLAP,
            separators    = ["\n\n", "\n", ".", " ", ""],
        )

    # ── Text ───────────────────────────────────────────────────────────────

    def _extract_text(self, pdf_path: str) -> list[Document]:
        """Extract plain text page-by-page using PyMuPDF and chunk it."""
        docs: list[Document] = []
        with fitz.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                raw = page.get_text("text").strip()
                if not raw:
                    continue
                for chunk_idx, chunk in enumerate(self.text_splitter.split_text(raw)):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "source"       : Path(pdf_path).name,
                            "page"         : page_num,
                            "chunk_index"  : chunk_idx,
                            "content_type" : "text",
                            "chunk_id"     : str(uuid.uuid4()),
                        },
                    ))
        log.info("Text extraction: %d chunks", len(docs))
        return docs

    # ── Tables ─────────────────────────────────────────────────────────────

    @staticmethod
    def _table_to_markdown(table: list[list[Any]]) -> str:
        """Convert a pdfplumber table (list-of-rows) to a Markdown string."""
        if not table or not table[0]:
            return ""
        header = table[0]
        rows   = table[1:]
        md  = "| " + " | ".join(str(c or "") for c in header) + " |\n"
        md += "| " + " | ".join(["---"] * len(header))         + " |\n"
        for row in rows:
            md += "| " + " | ".join(str(c or "") for c in row) + " |\n"
        return md

    def _extract_tables(self, pdf_path: str) -> list[Document]:
        """Extract all tables via pdfplumber and store as Markdown documents."""
        docs: list[Document] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for t_idx, table in enumerate(page.extract_tables()):
                    md = self._table_to_markdown(table)
                    if not md.strip():
                        continue
                    docs.append(Document(
                        page_content=f"[Table — page {page_num}, table {t_idx + 1}]\n{md}",
                        metadata={
                            "source"       : Path(pdf_path).name,
                            "page"         : page_num,
                            "table_index"  : t_idx,
                            "content_type" : "table",
                            "chunk_id"     : str(uuid.uuid4()),
                        },
                    ))
        log.info("Table extraction: %d tables", len(docs))
        return docs

    # ── Images ─────────────────────────────────────────────────────────────

    def _generate_caption(self, pil_image: Image.Image) -> str:
        """Run BLIP captioning on a PIL image and return the caption string."""
        inputs = self.caption_processor(pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.caption_model.generate(**inputs, max_new_tokens=128)
        return self.caption_processor.decode(out[0], skip_special_tokens=True)

    def _extract_images(self, pdf_path: str) -> list[Document]:
        """Extract embedded images, generate BLIP captions, return as Documents."""
        docs: list[Document] = []
        with fitz.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                for img_idx, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    try:
                        base_img  = pdf.extract_image(xref)
                        pil_image = Image.open(io.BytesIO(base_img["image"])).convert("RGB")
                        if pil_image.width < 50 or pil_image.height < 50:
                            continue
                        caption = self._generate_caption(pil_image)
                        docs.append(Document(
                            page_content=(
                                f"[Image - page {page_num}, image {img_idx + 1}]\n"
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
                        ))
                    except Exception as exc:
                        log.warning("Skipping image xref=%d page=%d: %s", xref, page_num, exc)
        log.info("Image extraction: %d captions", len(docs))
        return docs

    def extract(self, pdf_path: str) -> list[Document]:
        """Run all three extractors and return a unified document list."""
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
    """Wraps HuggingFace embeddings via langchain-huggingface."""

    def __init__(self, model_name: str = Config.EMBED_MODEL):
        self.model_name = model_name
        log.info("Loading embedding model: %s (device=%s)", model_name, Config.DEVICE)
        self.embeddings = HuggingFaceEmbeddings(
            model_name    = model_name,
            model_kwargs  = {"device": Config.DEVICE},
            encode_kwargs = {"normalize_embeddings": True},
        )

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Return the LangChain-compatible embedding object."""
        return self.embeddings


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 3 — VECTOR STORE MANAGER  (multi-PDF aware)
# ═══════════════════════════════════════════════════════════════════════════

class VectorStoreManager:
    """
    Manages a single in-memory ChromaDB collection that accumulates documents
    from ALL indexed PDFs. New PDFs are appended; nothing is ever wiped.
    """

    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.vectorstore: Chroma | None = None
        self._chroma_client = chromadb.EphemeralClient()

    def add_documents(self, documents: list[Document]) -> None:
        """Embed and ADD documents into the shared collection."""
        log.info("Adding %d documents to ChromaDB collection '%s'",
                 len(documents), Config.COLLECTION)
        if self.vectorstore is None:
            # First call — create the collection
            self.vectorstore = Chroma.from_documents(
                documents       = documents,
                embedding       = self.embedding_manager.get_embeddings(),
                client          = self._chroma_client,
                collection_name = Config.COLLECTION,
            )
        else:
            # Subsequent calls — append to existing collection
            self.vectorstore.add_documents(documents)
        log.info("ChromaDB updated — all indexed PDFs are searchable.")

    def get_retriever(self, k: int = Config.TOP_K):
        if self.vectorstore is None:
            raise RuntimeError("No documents indexed yet.")
        return self.vectorstore.as_retriever(
            search_type   = "similarity",
            search_kwargs = {"k": k},
        )

    def is_ready(self) -> bool:
        return self.vectorstore is not None


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 4 — GROQ LLM MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class GroqLLMManager:
    """Wraps Groq ChatGroq LLM with a ChatPromptTemplate chain."""

    CHAT_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert assistant that answers questions strictly based on the "
                "provided context. The context may contain text excerpts, table data (in "
                "Markdown), and image descriptions extracted from one or more PDF documents.\n\n"
                "Instructions:\n"
                "- Answer concisely and accurately using ONLY the context provided.\n"
                "- If the answer spans multiple PDFs, clearly state which document each piece "
                "of information comes from.\n"
                "- If the answer involves a table, reference the relevant rows and columns.\n"
                "- If the answer involves an image, reference the visual description.\n"
                "- If the context lacks enough information, say: "
                "'I don't have enough information to answer this.'"
            ),
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ])

    def __init__(self, api_key: str, model_name: str = Config.GROQ_MODEL):
        log.info("Initialising Groq LLM: %s", model_name)
        self.llm = ChatGroq(
            groq_api_key = api_key,
            model_name   = model_name,
            temperature  = 0.1,
            max_tokens   = 1024,
        )
        self.chain = self.CHAT_PROMPT | self.llm

    def generate(self, context: str, question: str) -> str:
        response = self.chain.invoke({"context": context, "question": question})
        answer   = response.content if hasattr(response, "content") else str(response)
        return answer.strip()


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 5 — RAG PIPELINE (Orchestrator — multi-PDF)
# ═══════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    End-to-end orchestrator with multi-PDF support.

    - A single VectorStoreManager accumulates documents from every PDF.
    - index_one(uploaded_file) adds one PDF; call repeatedly for more.
    - query() searches across ALL indexed PDFs simultaneously.
    - Duplicate filenames are automatically skipped.
    """

    def __init__(self, groq_api_key: str):
        self.embedding_manager    = _load_embedding_manager()   # cached loader
        self.vector_store_manager = VectorStoreManager(self.embedding_manager)
        self.llm_manager          = GroqLLMManager(api_key=groq_api_key)

        # Registry: filename -> doc count  (insertion-ordered dict)
        self.indexed_pdfs : dict[str, int] = {}

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            ctype  = doc.metadata.get("content_type", "text")
            page   = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "unknown")
            parts.append(
                f"[{i}] ({ctype.upper()} | {source} | Page {page})\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Single-PDF indexing ────────────────────────────────────────────────

    def index_one(self, uploaded_file, status_callback=None) -> int:
        """
        Parse ONE PDF and ADD its documents to the shared index.
        Returns 0 and skips silently if already indexed.
        """
        pdf_name = uploaded_file.name

        if pdf_name in self.indexed_pdfs:
            msg = f"'{pdf_name}' is already indexed — skipping."
            log.warning(msg)
            if status_callback:
                status_callback(f"skipped:{msg}")
            return 0

        def _log(msg: str):
            log.info(msg)
            if status_callback:
                status_callback(msg)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            cap_processor, cap_model = _load_caption_model()
            extractor = PDFExtractor(
                caption_processor = cap_processor,
                caption_model     = cap_model,
            )

            _log(f"[{pdf_name}] Extracting text...")
            text_docs = extractor._extract_text(tmp_path)
            _log(f"   {len(text_docs)} text chunks")

            _log(f"[{pdf_name}] Extracting tables...")
            table_docs = extractor._extract_tables(tmp_path)
            _log(f"   {len(table_docs)} tables")

            _log(f"[{pdf_name}] Captioning images...")
            if Config.DEVICE == "cpu":
                _log("   Running on CPU - image captioning may take a few minutes...")
            image_docs = extractor._extract_images(tmp_path)
            _log(f"   {len(image_docs)} image captions")

            new_docs = text_docs + table_docs + image_docs
            _log(f"[{pdf_name}] Total: {len(new_docs)} documents - adding to index...")

            self.vector_store_manager.add_documents(new_docs)
            self.indexed_pdfs[pdf_name] = len(new_docs)

            _log(f"[{pdf_name}] Done!")

        finally:
            os.unlink(tmp_path)

        return len(new_docs)

    # ── Querying ───────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """Retrieve and answer across ALL indexed PDFs."""
        if not self.vector_store_manager.is_ready():
            raise RuntimeError("No PDFs indexed. Call index_one() first.")

        retriever = self.vector_store_manager.get_retriever(k=Config.TOP_K)
        top_docs  = retriever.invoke(question)
        log.info("Similarity retrieval: %d documents", len(top_docs))

        context = self._build_context(top_docs)
        answer  = self.llm_manager.generate(context=context, question=question)

        sources = [
            {
                "content_type": d.metadata.get("content_type"),
                "page"        : d.metadata.get("page"),
                "source"      : d.metadata.get("source"),
            }
            for d in top_docs
        ]

        return {"answer": answer, "sources": sources, "top_docs": top_docs}

    def is_ready(self) -> bool:
        return self.vector_store_manager.is_ready()

    @property
    def pdf_count(self) -> int:
        return len(self.indexed_pdfs)

    @property
    def pdf_names(self) -> list[str]:
        return list(self.indexed_pdfs.keys())


# ═══════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model...")
def _load_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()


@st.cache_resource(show_spinner="Loading image captioning model (BLIP)...")
def _load_caption_model():
    dtype     = torch.float16 if Config.DEVICE == "cuda" else torch.float32
    processor = BlipProcessor.from_pretrained(Config.CAPTION_MODEL)
    model     = BlipForConditionalGeneration.from_pretrained(
        Config.CAPTION_MODEL, torch_dtype=dtype
    ).to(Config.DEVICE)
    return processor, model


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 6 — STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════

class StreamlitApp:
    """
    Streamlit UI — supports uploading and querying multiple PDFs in one session.
    """

    BADGE = {"text": "📝", "table": "📊", "image": "🖼️"}

    def __init__(self):
        st.set_page_config(
            page_title = "Multimodal PDF RAG",
            page_icon  = "📚",
            layout     = "wide",
        )
        self._init_session_state()

    # ── Session state ──────────────────────────────────────────────────────

    @staticmethod
    def _init_session_state():
        defaults = {
            "pipeline"     : None,
            "chat_history" : [],
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # ── Sidebar ────────────────────────────────────────────────────────────

    def _render_sidebar(self) -> tuple[str, list, bool]:
        with st.sidebar:
            st.title("📚 Multimodal PDF RAG")
            st.caption("Text · Tables · Images - Groq LLM")
            st.divider()

            groq_key = st.text_input(
                "🔑 Groq API Key",
                type  = "password",
                value = st.secrets.get("GROQ_API_KEY", ""),
                help  = "Free key at https://console.groq.com",
            )

            st.divider()

            # ── Multi-file uploader ────────────────────────────────────────
            uploaded_files = st.file_uploader(
                "📄 Upload PDF(s)",
                type                  = ["pdf"],
                accept_multiple_files = True,
                help                  = "Select one or more PDFs. Already-indexed files are skipped automatically.",
            )

            index_clicked = st.button(
                "⚡ Index PDF(s)",
                disabled            = (not uploaded_files or not groq_key),
                use_container_width = True,
                type                = "primary",
            )

            # ── Indexed PDFs panel ─────────────────────────────────────────
            pipeline: RAGPipeline | None = st.session_state.pipeline
            if pipeline and pipeline.pdf_count > 0:
                st.divider()
                st.markdown(f"**📂 Indexed PDFs ({pipeline.pdf_count})**")
                for name, count in pipeline.indexed_pdfs.items():
                    st.markdown(f"- `{name}`  ({count} docs)")

            # ── Clear all button ───────────────────────────────────────────
            if pipeline and pipeline.pdf_count > 0:
                st.divider()
                if st.button("🗑️ Clear all & start over", use_container_width=True):
                    st.session_state.pipeline     = None
                    st.session_state.chat_history = []
                    st.rerun()

            st.divider()
            st.markdown("""
**Stack**
- 🤗 Embeddings : `all-MiniLM-L6-v2`
- 🔎 Retrieval  : ChromaDB Similarity Search
- 🖼️ Captions   : `BLIP`
- 🗄️ Store      : ChromaDB (in-memory)
- 🚀 LLM        : Groq `llama-3.3-70b-versatile`
            """)

        return groq_key, uploaded_files, index_clicked

    # ── Indexing flow ──────────────────────────────────────────────────────

    def _handle_indexing(self, groq_key: str, uploaded_files: list) -> None:
        """Create or reuse the RAGPipeline; index every uploaded file."""
        if st.session_state.pipeline is None:
            st.session_state.pipeline = RAGPipeline(groq_api_key=groq_key)

        pipeline: RAGPipeline = st.session_state.pipeline

        for uploaded_file in uploaded_files:
            pdf_name = uploaded_file.name

            if pdf_name in pipeline.indexed_pdfs:
                st.info(f"ℹ️ **{pdf_name}** is already indexed — skipped.")
                continue

            messages: list[str] = []

            def status_callback(msg: str):
                messages.append(msg)

            with st.status(f"Processing **{pdf_name}**...", expanded=True) as status_box:
                total = pipeline.index_one(uploaded_file,
                                           status_callback=status_callback)
                for msg in messages:
                    # Filter out the internal skip prefix
                    if not msg.startswith("skipped:"):
                        status_box.write(msg)
                if total > 0:
                    status_box.update(
                        label = f"✅ Indexed {total} documents from **{pdf_name}**",
                        state = "complete",
                    )

        names = ", ".join(f"**{n}**" for n in pipeline.pdf_names)
        st.success(f"✅ Active knowledge base: {names}")

    # ── Chat history ───────────────────────────────────────────────────────

    def _render_chat_history(self) -> None:
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])
                self._render_sources(turn.get("sources", []))

    def _render_sources(self, sources: list[dict]) -> None:
        if not sources:
            return
        with st.expander("📎 Sources used"):
            for s in sources:
                badge  = self.BADGE.get(s.get("content_type", ""), "📄")
                ctype  = (s.get("content_type") or "").upper()
                source = s.get("source", "")
                page   = s.get("page", "?")
                st.markdown(f"{badge} **{ctype}** | `{source}` | Page {page}")

    # ── Query flow ─────────────────────────────────────────────────────────

    def _handle_query(self, question: str) -> None:
        pipeline: RAGPipeline = st.session_state.pipeline

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching across all PDFs and generating answer..."):
                result = pipeline.query(question)
            st.write(result["answer"])
            self._render_sources(result["sources"])

        st.session_state.chat_history.append({
            "question": question,
            "answer"  : result["answer"],
            "sources" : result["sources"],
        })

    # ── Main entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        groq_key, uploaded_files, index_clicked = self._render_sidebar()

        if index_clicked and uploaded_files and groq_key:
            self._handle_indexing(groq_key, uploaded_files)

        pipeline: RAGPipeline | None = st.session_state.pipeline
        if pipeline is None or not pipeline.is_ready():
            st.info("👈 Upload one or more PDFs and click **Index PDF(s)** to get started.")
            return

        # ── Chat header ────────────────────────────────────────────────────
        if pipeline.pdf_count == 1:
            st.header(f"💬 Chat with: `{pipeline.pdf_names[0]}`")
        else:
            st.header(f"💬 Chat across {pipeline.pdf_count} PDFs")
            cols = st.columns(min(pipeline.pdf_count, 4))
            for col, name in zip(cols, pipeline.pdf_names):
                col.markdown(f"📄 `{name}`")

        self._render_chat_history()

        question = st.chat_input(
            "Ask anything — answers draw from all indexed PDFs..."
        )
        if question:
            if not groq_key:
                st.warning("Please enter your Groq API key in the sidebar.")
                return
            self._handle_query(question)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    StreamlitApp().run()
