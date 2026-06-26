"""
FastAPI Backend for Agentic RAG Knowledge Assistant.

Endpoints:
    - POST /upload   : Upload and index PDF documents.
    - POST /query    : Submit a query and get an AI-generated response.
    - GET  /health   : Health check endpoint.
"""

import sys
import os
import tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.config.settings import Config
from src.ingestion.pdf_loader import PDFLoader
from src.embedding.embedding import EmbeddingManager
from src.vectorstore.chroma_store import ChromaVectorStore
from src.tools.retriever_tool import RetrieverTool
from src.graph.workflow_graph import GraphBuilder
import warnings

warnings.filterwarnings("ignore")

# ─── Mock Streamlit session_state for FastAPI context ──────────────────────────
# The Agent and tools use st.session_state to access vector_retriever and user_controls.
# We need to ensure this works outside of a Streamlit runtime.
import streamlit as st

if not hasattr(st, "session_state") or st.session_state is None:
    st.session_state = {}

# Initialize required keys
if "user_controls" not in st.session_state:
    st.session_state["user_controls"] = {}
if "vector_retriever" not in st.session_state:
    st.session_state["vector_retriever"] = None


# ─── Pydantic Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str
    llm_model: Optional[str] = None
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 800


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    tool_used: Optional[str] = None


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    message: str
    pages_extracted: int
    files_processed: List[str]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    documents_indexed: bool


# ─── Application State ─────────────────────────────────────────────────────────

class AppState:
    """
    In-memory application state for the FastAPI backend.

    Stores:
        - Vector retriever (after document indexing)
        - User configuration settings
        - Graph builder instance
    """

    def __init__(self):
        self.vector_retriever = None
        self.config = Config()
        self.graph_builder = GraphBuilder()
        self.user_controls = {
            "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
            "LLM_MODEL": "llama-3.3-70b-versatile",
            "TEMPERATURE": 0.2,
            "TOKEN": 800,
            "CHUNK_SIZE": self.config.get_chunk_size(),
            "CHUNK_OVERLAP": self.config.get_chunk_overlap(),
            "TOP_K": self.config.get_top_k(),
            "EMBEDDING_MODELS": self.config.get_embedding_model(),
            "CAPTION_MODEL": self.config.get_caption_model(),
        }


# ─── FastAPI App Initialization ───────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG Knowledge Assistant API",
    description="FastAPI backend for Agentic RAG with document upload and query capabilities.",
    version="1.0.0",
)

# CORS middleware for Streamlit frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global application state
state = AppState()


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        documents_indexed=state.vector_retriever is not None,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    llm_model: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    max_tokens: Optional[int] = Form(None),
    groq_api_key: Optional[str] = Form(None),
):
    """
    Upload and index PDF documents.

    Steps:
        1. Validate uploaded files are PDFs.
        2. Load and chunk documents.
        3. Generate embeddings.
        4. Create vector store.
        5. Configure retriever.

    Args:
        files: List of PDF files to upload.
        llm_model: Optional LLM model name.
        temperature: Optional temperature setting.
        max_tokens: Optional max tokens setting.
        groq_api_key: Optional Groq API key.

    Returns:
        UploadResponse with processing details.
    """
    try:
        logging.info("API: Upload request received with %d file(s).", len(files))

        # Update user controls if provided
        if llm_model:
            state.user_controls["LLM_MODEL"] = llm_model
        if temperature is not None:
            state.user_controls["TEMPERATURE"] = temperature
        if max_tokens is not None:
            state.user_controls["TOKEN"] = max_tokens
        if groq_api_key:
            state.user_controls["GROQ_API_KEY"] = groq_api_key

        all_documents = []
        processed_files = []

        for file in files:
            if file.content_type != "application/pdf":
                logging.warning("Skipping non-PDF file: %s", file.filename)
                continue

            try:
                # Read file content
                content = await file.read()

                # Create a file-like object for PDFLoader
                uploaded_file = UploadedFileWrapper(
                    content=content,
                    name=file.filename,
                    file_type="application/pdf",
                )

                loader = PDFLoader(uploaded_file, state.user_controls)
                documents = loader.load_documents()
                all_documents.extend(documents)
                processed_files.append(file.filename)

                logging.info(
                    "PDF loaded | File=%s | Pages=%d",
                    file.filename,
                    len(documents),
                )

            except Exception as e:
                logging.error("Failed to load %s: %s", file.filename, str(e))
                continue

        if not all_documents:
            raise HTTPException(
                status_code=400,
                detail="No valid PDF documents could be processed.",
            )

        # Create Embeddings
        embedding_mgr = EmbeddingManager(state.user_controls["EMBEDDING_MODELS"])
        embeddings = embedding_mgr.create_embeddings()
        logging.info("API: Embeddings created successfully.")

        # Create Vector Store
        vector_store_mgr = ChromaVectorStore(all_documents, embeddings)
        vector_db = vector_store_mgr.create_vectorstore()
        logging.info("API: Vector store created successfully.")

        # Create Retriever
        retriever_mgr = RetrieverTool(vector_db, top_k=state.user_controls["TOP_K"])
        vector_retriever = retriever_mgr.get_retriever()

        # Store retriever in app state
        state.vector_retriever = vector_retriever

        # Synchronize with st.session_state for tool compatibility
        st.session_state["vector_retriever"] = vector_retriever
        st.session_state["user_controls"] = state.user_controls

        logging.info("API: Retriever stored in application state.")

        return UploadResponse(
            message="Documents indexed successfully.",
            pages_extracted=len(all_documents),
            files_processed=processed_files,
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("API: Upload endpoint failed.")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Process a user query through the Agentic RAG pipeline.

    Args:
        request: QueryRequest with the user's question and optional settings.

    Returns:
        QueryResponse with the AI-generated answer and tool used.
    """
    try:
        logging.info("API: Query received: '%s'", request.question)

        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        # Update user controls if provided
        if request.llm_model:
            state.user_controls["LLM_MODEL"] = request.llm_model
        if request.temperature is not None:
            state.user_controls["TEMPERATURE"] = request.temperature
        if request.max_tokens is not None:
            state.user_controls["MAX_TOKENS"] = request.max_tokens

        # Ensure GROQ API key is available
        groq_api_key = state.user_controls.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            raise HTTPException(
                status_code=400,
                detail="GROQ_API_KEY not configured. Please provide it via upload endpoint or environment variable.",
            )
        state.user_controls["GROQ_API_KEY"] = groq_api_key

        # Synchronize state with st.session_state for Agent and tools compatibility
        st.session_state["user_controls"] = state.user_controls
        st.session_state["vector_retriever"] = state.vector_retriever

        # Build and execute graph
        graph = state.graph_builder.build_graph()
        response, tool_name = state.graph_builder.execute(graph, request.question.strip())

        logging.info("API: Query processed. Tool used: %s", tool_name or "None")

        return QueryResponse(
            answer=response if response else "No response generated.",
            tool_used=tool_name if tool_name else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("API: Query endpoint failed.")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# ─── Helper Classes ────────────────────────────────────────────────────────────

class UploadedFileWrapper:
    """
    Wrapper class to mimic Streamlit's UploadedFile interface.

    This allows the existing PDFLoader to work with FastAPI file uploads
    without modification.
    """

    def __init__(self, content: bytes, name: str, file_type: str):
        self._content = content
        self.name = name
        self.type = file_type
        self.size = len(content)
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            data = self._content[self._position:]
            self._position = len(self._content)
        else:
            data = self._content[self._position:self._position + size]
            self._position += size
        return data

    def seek(self, position: int) -> None:
        self._position = position

    def tell(self) -> int:
        return self._position

    def getvalue(self) -> bytes:
        return self._content
