"""
Multimodal RAG Pipeline — Streamlit Cloud App (Class-Based Architecture)
=========================================================================
Open-source stack:
  - PDF parsing   : PyMuPDF (fitz) + pdfplumber
  - Image caption : BLIP  (Salesforce/blip-image-captioning-base)
  - Embeddings    : HuggingFace BGE-M3  (BAAI/bge-m3)  via langchain-huggingface
  - Sparse search : BM25  (rank_bm25)  via LangChain BM25Retriever
  - Reranker      : BGE-Reranker-v2-M3 (BAAI/bge-reranker-v2-m3)
  - Vector store  : ChromaDB  (EphemeralClient — in-memory per session)
  - LLM           : Groq  (llama-3.1-8b-instant — free tier)
  - Framework     : LangChain + Streamlit

Secrets (Streamlit Cloud → Settings → Secrets):
    GROQ_API_KEY = "gsk_..."

Local run:
    pip install -r requirements.txt
    streamlit run app.py

Architecture (classes):
    PDFExtractor          — text / table / image extraction from PDF bytes
    EmbeddingManager      — HuggingFace BGE-M3 embedding wrapper
    VectorStoreManager    — ChromaDB in-memory store + dense retriever
    HybridRetrieverBuilder— EnsembleRetriever (dense BM25 fusion via RRF)
    Reranker              — BGE cross-encoder reranker
    GroqLLMManager        — Groq ChatGroq LLM wrapper
    RAGPipeline           — orchestrator: index + query
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
from sentence_transformers import CrossEncoder

# ── LangChain ──────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
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
    EMBED_MODEL    : str = "BAAI/bge-m3"
    RERANKER_MODEL : str = "BAAI/bge-reranker-v2-m3"
    CAPTION_MODEL  : str = "Salesforce/blip-image-captioning-base"
    GROQ_MODEL     : str = "llama-3.1-8b-instant"   # free-tier Groq

    # Chunking
    CHUNK_SIZE     : int = 800
    CHUNK_OVERLAP  : int = 150

    # Retrieval
    TOP_K_DENSE    : int = 10   # candidates from ChromaDB
    TOP_K_SPARSE   : int = 10   # candidates from BM25
    TOP_K_RERANK   : int = 5    # final docs passed to LLM
    DENSE_WEIGHT   : float = 0.6
    SPARSE_WEIGHT  : float = 0.4

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
                        # Skip decorative / tiny images
                        if pil_image.width < 50 or pil_image.height < 50:
                            continue
                        caption = self._generate_caption(pil_image)
                        docs.append(Document(
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
                        ))
                    except Exception as exc:
                        log.warning("Skipping image xref=%d page=%d: %s", xref, page_num, exc)
        log.info("Image extraction: %d captions", len(docs))
        return docs

    # ── Public API ─────────────────────────────────────────────────────────

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
    """
    Wraps HuggingFace BGE-M3 embeddings via langchain-huggingface.
    Provides a ready-to-use LangChain embedding object.
    """

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
# CLASS 3 — VECTOR STORE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class VectorStoreManager:
    """
    Manages an in-memory ChromaDB collection.
    Exposes a LangChain dense retriever for downstream use.
    """

    def __init__(self, embedding_manager: EmbeddingManager, collection_name: str = "multimodal_rag"):
        self.embedding_manager = embedding_manager
        self.collection_name   = collection_name
        self.vectorstore: Chroma | None = None
        # EphemeralClient → in-memory, no disk writes, works on Streamlit Cloud
        self._chroma_client = chromadb.EphemeralClient()

    def index_documents(self, documents: list[Document]) -> None:
        """Embed and store all documents into ChromaDB."""
        log.info("Indexing %d documents into ChromaDB…", len(documents))
        self.vectorstore = Chroma.from_documents(
            documents       = documents,
            embedding       = self.embedding_manager.get_embeddings(),
            client          = self._chroma_client,
            collection_name = self.collection_name,
        )
        log.info("ChromaDB indexing complete — collection: %s", self.collection_name)

    def get_dense_retriever(self, k: int = Config.TOP_K_DENSE):
        """Return a LangChain retriever backed by ChromaDB similarity search."""
        if self.vectorstore is None:
            raise RuntimeError("Call index_documents() before get_dense_retriever().")
        return self.vectorstore.as_retriever(
            search_type   = "similarity",
            search_kwargs = {"k": k},
        )

    def is_ready(self) -> bool:
        return self.vectorstore is not None


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 4 — HYBRID RETRIEVER BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class HybridRetrieverBuilder:
    """
    Combines a dense ChromaDB retriever with a sparse BM25 retriever
    using LangChain's EnsembleRetriever (Reciprocal Rank Fusion).
    """

    def __init__(
        self,
        dense_weight : float = Config.DENSE_WEIGHT,
        sparse_weight: float = Config.SPARSE_WEIGHT,
        sparse_k     : int   = Config.TOP_K_SPARSE,
    ):
        self.dense_weight  = dense_weight
        self.sparse_weight = sparse_weight
        self.sparse_k      = sparse_k

    def build(
        self,
        documents      : list[Document],
        dense_retriever,
    ) -> EnsembleRetriever:
        """
        Build and return an EnsembleRetriever.

        Args:
            documents:       Full document list (needed by BM25).
            dense_retriever: ChromaDB-backed LangChain retriever.
        """
        log.info(
            "Building hybrid retriever: dense=%.1f, sparse=%.1f, sparse_k=%d",
            self.dense_weight, self.sparse_weight, self.sparse_k,
        )
        bm25_retriever   = BM25Retriever.from_documents(documents)
        bm25_retriever.k = self.sparse_k

        return EnsembleRetriever(
            retrievers = [dense_retriever, bm25_retriever],
            weights    = [self.dense_weight, self.sparse_weight],
        )


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 5 — RERANKER
# ═══════════════════════════════════════════════════════════════════════════

class Reranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.
    Scores every (query, document) pair and returns the top-k.
    """

    def __init__(self, model_name: str = Config.RERANKER_MODEL):
        log.info("Loading reranker: %s (device=%s)", model_name, Config.DEVICE)
        self.model = CrossEncoder(model_name, max_length=512, device=Config.DEVICE)

    def rerank(
        self,
        query    : str,
        documents: list[Document],
        top_k    : int = Config.TOP_K_RERANK,
    ) -> list[Document]:
        """Score all candidate documents and return the highest-scoring top_k."""
        if not documents:
            return []
        pairs  = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        result = [doc for _, doc in ranked[:top_k]]
        log.info("Reranker: %d → top %d", len(documents), len(result))
        return result


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 6 — GROQ LLM MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class GroqLLMManager:
    """
    Wraps the Groq ChatGroq LLM (llama-3.1-8b-instant, free tier).
    Provides a generate() method that accepts a formatted prompt string.
    """

    # RAG prompt template (shared across all instances)
    PROMPT_TEMPLATE = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert assistant that answers questions strictly based on the provided context.
The context may contain text excerpts, table data (in Markdown), and image descriptions extracted from a PDF.

Context:
{context}

Question: {question}

Instructions:
- Answer concisely and accurately using ONLY the context above.
- If the answer involves a table, reference the relevant rows and columns.
- If the answer involves an image, reference the visual description.
- If the context lacks enough information, say "I don't have enough information to answer this."

Answer:""",
    )

    def __init__(self, api_key: str, model_name: str = Config.GROQ_MODEL):
        log.info("Initialising Groq LLM: %s", model_name)
        self.llm = ChatGroq(
            groq_api_key = api_key,
            model_name   = model_name,
            temperature  = 0.1,
            max_tokens   = 1024,
        )

    def generate(self, context: str, question: str) -> str:
        """Format the RAG prompt and call the Groq API; return the answer string."""
        prompt   = self.PROMPT_TEMPLATE.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        answer   = response.content if hasattr(response, "content") else str(response)
        return answer.strip()


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 7 — RAG PIPELINE (Orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    End-to-end orchestrator:
      PDF upload → PDFExtractor → EmbeddingManager → VectorStoreManager
               → HybridRetrieverBuilder → Reranker → GroqLLMManager → Answer

    Usage:
        pipeline = RAGPipeline(groq_api_key="gsk_...")
        pipeline.index(uploaded_file, status_callback)
        result   = pipeline.query("What does the table show?")
    """

    def __init__(self, groq_api_key: str):
        # Instantiate all components
        self.embedding_manager    = EmbeddingManager()
        self.vector_store_manager = VectorStoreManager(self.embedding_manager)
        self.hybrid_builder       = HybridRetrieverBuilder()
        self.reranker             = Reranker()
        self.llm_manager          = GroqLLMManager(api_key=groq_api_key)

        # Will be populated after indexing
        self.documents : list[Document] = []
        self.pdf_name  : str            = ""

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(docs: list[Document]) -> str:
        """Format top-k reranked documents into a single context block."""
        parts = []
        for i, doc in enumerate(docs, 1):
            ctype  = doc.metadata.get("content_type", "text")
            page   = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "unknown")
            parts.append(
                f"[{i}] ({ctype.upper()} | {source} | Page {page})\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Indexing ───────────────────────────────────────────────────────────

    def index(self, uploaded_file, status_callback=None) -> int:
        """
        Parse the uploaded Streamlit file and index all content into ChromaDB.

        Args:
            uploaded_file:   Streamlit UploadedFile object.
            status_callback: Optional callable(str) for progress messages.

        Returns:
            Total number of indexed documents.
        """
        def _log(msg: str):
            log.info(msg)
            if status_callback:
                status_callback(msg)

        self.pdf_name  = uploaded_file.name
        self.documents = []

        # Save to temp file (PyMuPDF / pdfplumber need a real path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Load BLIP lazily here so the cached resource is reused
            cap_processor, cap_model = _load_caption_model()
            extractor = PDFExtractor(
                caption_processor = cap_processor,
                caption_model     = cap_model,
            )

            _log("📄 Extracting text…")
            text_docs = extractor._extract_text(tmp_path)
            _log(f"   ✅ {len(text_docs)} text chunks")

            _log("📊 Extracting tables…")
            table_docs = extractor._extract_tables(tmp_path)
            _log(f"   ✅ {len(table_docs)} tables")

            _log("🖼️ Captioning images…")
            image_docs = extractor._extract_images(tmp_path)
            _log(f"   ✅ {len(image_docs)} image captions")

            self.documents = text_docs + table_docs + image_docs
            _log(f"📦 Total: {len(self.documents)} documents")

            _log("🔢 Building ChromaDB vector index…")
            self.vector_store_manager.index_documents(self.documents)
            _log("✅ Indexing complete!")

        finally:
            os.unlink(tmp_path)

        return len(self.documents)

    # ── Querying ───────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """
        Run the full retrieval-generation pipeline for a user question.

        Returns:
            {
              "answer" : str,
              "sources": list[dict],   # content_type, page, source
              "top_docs": list[Document]
            }
        """
        if not self.vector_store_manager.is_ready():
            raise RuntimeError("Pipeline not indexed. Call index() first.")
        if not self.documents:
            raise RuntimeError("Document list is empty; cannot build BM25 index.")

        # Step 1 — Hybrid retrieval (dense + BM25)
        dense_retriever  = self.vector_store_manager.get_dense_retriever(k=Config.TOP_K_DENSE)
        hybrid_retriever = self.hybrid_builder.build(
            documents       = self.documents,
            dense_retriever = dense_retriever,
        )
        candidates = hybrid_retriever.invoke(question)
        log.info("Hybrid retrieval returned %d candidates", len(candidates))

        # Step 2 — Rerank
        top_docs = self.reranker.rerank(question, candidates, top_k=Config.TOP_K_RERANK)

        # Step 3 — Build context and generate answer via Groq
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

        return {
            "answer"   : answer,
            "sources"  : sources,
            "top_docs" : top_docs,
        }

    def is_ready(self) -> bool:
        """Return True once a PDF has been indexed."""
        return self.vector_store_manager.is_ready()


# ═══════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADERS  (Streamlit — loaded once per session)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model (BGE-M3)…")
def _load_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()


@st.cache_resource(show_spinner="Loading reranker (BGE-Reranker-v2-M3)…")
def _load_reranker() -> Reranker:
    return Reranker()


@st.cache_resource(show_spinner="Loading image captioning model (BLIP)…")
def _load_caption_model():
    """Returns (BlipProcessor, BlipForConditionalGeneration) — cached once."""
    dtype     = torch.float16 if Config.DEVICE == "cuda" else torch.float32
    processor = BlipProcessor.from_pretrained(Config.CAPTION_MODEL)
    model     = BlipForConditionalGeneration.from_pretrained(
        Config.CAPTION_MODEL, torch_dtype=dtype
    ).to(Config.DEVICE)
    return processor, model


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 8 — STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════

class StreamlitApp:
    """
    Encapsulates all Streamlit UI logic.
    Call StreamlitApp().run() as the entry point.
    """

    # Content-type → emoji badge
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
            "pipeline"     : None,   # RAGPipeline instance
            "chat_history" : [],     # list of {question, answer, sources}
            "pdf_name"     : "",
            "groq_api_key" : "",
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # ── Sidebar ────────────────────────────────────────────────────────────

    def _render_sidebar(self) -> tuple[str, Any, bool]:
        """
        Render sidebar widgets.
        Returns (groq_api_key, uploaded_file, index_clicked).
        """
        with st.sidebar:
            st.title("📚 Multimodal PDF RAG")
            st.caption("Text · Tables · Images → Groq LLM")
            st.divider()
            
            # Groq API key input
            groq_key = st.text_input(
                "🔑 Groq API Key",
                type  = "password",
                value = st.secrets.get("GROQ_API_KEY", ""),
                help  = "Free key at https://console.groq.com",
            )

            st.divider()

            # PDF uploader
            uploaded_file = st.file_uploader(
                "📄 Upload PDF",
                type = ["pdf"],
                help = "Any PDF with text, tables, or images.",
            )

            index_clicked = st.button(
                "⚡ Index PDF",
                disabled         = (uploaded_file is None or not groq_key),
                use_container_width = True,
                type             = "primary",
            )

            st.divider()
            st.markdown("""
**Stack**
- 🤗 Embeddings : `BAAI/bge-m3`
- 🔎 Retrieval  : Dense (ChromaDB) + BM25
- 🏆 Reranker   : `BGE-Reranker-v2-M3`
- 🖼️ Captions   : `BLIP`
- 🗄️ Store      : ChromaDB (in-memory)
- 🚀 LLM        : Groq `llama-3.1-8b-instant`
            """)

        return groq_key, uploaded_file, index_clicked

    # ── Indexing flow ──────────────────────────────────────────────────────

    def _handle_indexing(self, groq_key: str, uploaded_file) -> None:
        """Instantiate RAGPipeline and run indexing with a live status panel."""
        # Reset state for a fresh document
        st.session_state.pipeline     = None
        st.session_state.chat_history = []
        st.session_state.pdf_name     = uploaded_file.name

        pipeline = RAGPipeline(groq_api_key=groq_key)

        messages: list[str] = []

        def status_callback(msg: str):
            messages.append(msg)

        with st.status("Processing PDF…", expanded=True) as status_box:
            total = pipeline.index(uploaded_file, status_callback=status_callback)
            for msg in messages:
                status_box.write(msg)
            status_box.update(
                label = f"✅ Indexed {total} documents from **{uploaded_file.name}**",
                state = "complete",
            )

        st.session_state.pipeline = pipeline
        st.success(f"✅ **{uploaded_file.name}** is ready — ask your first question below!")

    # ── Chat history ───────────────────────────────────────────────────────

    def _render_chat_history(self) -> None:
        """Re-render all previous turns from session state."""
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])
                self._render_sources(turn.get("sources", []))

    def _render_sources(self, sources: list[dict]) -> None:
        """Render a collapsible sources panel beneath an answer."""
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
        """Run the RAG pipeline for a user question and render the result."""
        pipeline: RAGPipeline = st.session_state.pipeline

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer…"):
                result = pipeline.query(question)

            st.write(result["answer"])
            self._render_sources(result["sources"])

        # Persist in chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer"  : result["answer"],
            "sources" : result["sources"],
        })

    # ── Main entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        """Render the full Streamlit app."""
        groq_key, uploaded_file, index_clicked = self._render_sidebar()

        # ── Indexing triggered ────────────────────────────────────────────
        if index_clicked and uploaded_file and groq_key:
            self._handle_indexing(groq_key, uploaded_file)

        # ── Nothing indexed yet ───────────────────────────────────────────
        if st.session_state.pipeline is None:
            st.info("👈 Upload a PDF and click **Index PDF** to get started.")
            return

        # ── Chat interface ────────────────────────────────────────────────
        st.header(f"💬 Chat with: `{st.session_state.pdf_name}`")
        self._render_chat_history()

        question = st.chat_input("Ask anything about the PDF…")
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