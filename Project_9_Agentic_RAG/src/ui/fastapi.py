import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from src.ingestion.pdf_loader import PDFLoader
from src.embedding.embedding import EmbeddingManager
from src.vectorstore.chroma_store import ChromaVectorStore
from src.tools.retriever_tool import RetrieverTool
from src.graph.workflow_graph import GraphBuilder
import base64
import warnings
warnings.filterwarnings("ignore")

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    user_controls: dict = {}

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

# In-process, per-session store for retrievers.
#
# NOTE: This replaces the previous (broken) use of `st.session_state`
# inside the FastAPI process. `st.session_state` only exists inside an
# active Streamlit ScriptRunContext -- it is NOT available here, since
# FastAPI runs in its own thread/process with no such context. Using it
# here would raise at runtime / silently no-op.
#
# This dict is still an in-memory, single-process store: it is fine for
# the current single-worker deployment model described in the README,
# but it will NOT survive a process restart and will NOT be shared
# across multiple uvicorn workers/replicas. For true production use,
# back this with Redis (or another shared store) keyed by session_id.
SESSION_STORE: Dict[str, Dict[str, Any]] = {}

def _get_session(session_id: str) -> Dict[str, Any]:
    """Return (creating if necessary) the state dict for a session_id."""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = {}
    return SESSION_STORE[session_id]

class FastAPIState:
    def __init__(self):
        self._graph_builder = GraphBuilder()
        # Compile graph once at startup
        self._compiled_graph = self._graph_builder.build_graph()
        logging.info("Graph compiled once at FastAPI startup.")
        # Cache graph image as base64 for frontend rendering
        self._graph_image_b64 = None
        try:
            graph_png_bytes = self._compiled_graph.get_graph().draw_mermaid_png()
            self._graph_image_b64 = base64.b64encode(graph_png_bytes).decode("utf-8")
            logging.info("Graph PNG image generated and cached.")
        except Exception as e:
            logging.warning("Failed to generate graph PNG: %s", str(e))

app = FastAPI()

state = FastAPIState()

# Endpoints
@app.get("/health", response_model = HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status = "healthy")

@app.get("/graph")
async def get_graph():
    """Return the compiled graph image as base64 for frontend rendering."""
    return {"graph_image": state._graph_image_b64}

@app.post("/upload", response_model = UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: str = Form("default"),
    LLM_MODEL: str = Form(None),
    GROQ_API_KEY: str = Form(None),
    TEMPERATURE: float = Form(0.1),
    TOKEN: int = Form(2048),
    CHUNK_SIZE: int = Form(800),
    CHUNK_OVERLAP: int = Form(150),
    TOP_K: int = Form(5),
    EMBEDDING_MODELS: str = Form("all-MiniLM-L6-v2"),
    CAPTION_MODEL: str = Form("Salesforce/blip-image-captioning-base"),
):
    """
    Upload and index one or more PDF documents.

    NOTE: this endpoint previously declared `user_controls: dict` as a
    second parameter alongside `files: List[UploadFile]`. FastAPI cannot
    bind an arbitrary `dict` body parameter together with multipart file
    fields sent via `requests.post(files=..., data=...)` -- multipart
    form fields must be declared individually via `Form(...)`. That
    mismatch caused this endpoint to fail validation (422) or silently
    receive untyped/incorrect values (e.g. `TOP_K` arriving as a string).

    This version declares each expected field explicitly and typed, using
    the SAME field names (`LLM_MODEL`, `GROQ_API_KEY`, `TOP_K`, etc.) that
    the Streamlit frontend's `user_controls` dict already uses (see
    `src/ui/streamlit_app.py`), so no frontend changes are required.
    """
    try:
        # Re-assemble a typed user_controls dict for downstream components
        # that expect this shape (PDFLoader, EmbeddingManager, RetrieverTool).
        user_controls = {
            "LLM_MODEL": LLM_MODEL,
            "GROQ_API_KEY": GROQ_API_KEY,
            "TEMPERATURE": TEMPERATURE,
            "TOKEN": TOKEN,
            "CHUNK_SIZE": CHUNK_SIZE,
            "CHUNK_OVERLAP": CHUNK_OVERLAP,
            "TOP_K": TOP_K,
            "EMBEDDING_MODELS": EMBEDDING_MODELS,
            "CAPTION_MODEL": CAPTION_MODEL,
        }

        all_documents = []
        processed_files = []

        # Step 1: Load PDFs
        for file in files:
            try:
                loader = PDFLoader(file, user_controls)
                documents = loader.load_documents()
                all_documents.extend(documents)
                processed_files.append(file.filename)
                # Fix: UploadFile has `.filename`, not `.name`.
                logging.info("PDF loaded successfully | File=%s | Pages=%s", file.filename, len(documents))
            except Exception as e:
                # Fix: use `.filename` (not `.name`) and remove the
                # `st.error(...)` call -- Streamlit has no active script
                # context inside the FastAPI process, so this call was
                # invalid here. Errors are reported back via the HTTP
                # response / logs instead.
                logging.error("Failed to load %s: %s", file.filename, e)
                continue

        if not all_documents:
            raise HTTPException(status_code = 400, detail = "No valid PDF documents could be processed")

        # Step 2: Create Embeddings
        embedding_mgr = EmbeddingManager(user_controls["EMBEDDING_MODELS"])
        embeddings = embedding_mgr.create_embeddings()
        logging.info("Embeddings created successfully.")

        # Step 3: Create Vector Store
        vector_store_mgr = ChromaVectorStore(all_documents, embeddings)
        vector_db = vector_store_mgr.create_vectorstore()
        logging.info("Chroma vector store created successfully.")

        # Step 4: Create Retriever
        retriever_mgr = RetrieverTool(vector_db, top_k = user_controls["TOP_K"])
        vector_retriever = retriever_mgr.get_retriever()

        # Step 5: Store Retriever (per-session, not st.session_state)
        session = _get_session(session_id)
        session["vector_retriever"] = vector_retriever
        logging.info("Retriever stored for session_id=%s.", session_id)

        logging.info("Document ingestion pipeline completed successfully.")

        return UploadResponse(
            message = "Documents indexed successfully.",
            pages_extracted = len(all_documents),
            files_processed = processed_files
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Document ingestion pipeline failed.")
        raise HTTPException(status_code = 500, detail = f"Document processing failed: {str(e)}")

@app.post("/query", response_model = QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        logging.info("API: Query received: '%s'", request.question)

        if not request.question.strip():
            raise HTTPException(status_code = 400, detail = "Question cannot be empty")

        # Look up this session's retriever (if any).
        #
        # IMPORTANT: do NOT put the retriever object itself into
        # `user_controls` / graph state. Now that the graph is compiled
        # with a checkpointer (see GraphBuilder.build_graph), LangGraph
        # serializes the ENTIRE state -- including user_controls -- on
        # every turn so it can be persisted/reloaded by thread_id. A live
        # Chroma retriever object is not serializable, and would raise
        # `TypeError: ... is not msgpack serializable` at runtime.
        #
        # Instead, the retriever is set directly on RetrieverTool, which
        # the vector_db_retriever tool function reads back from when it
        # runs (see src/tools/retriever_tool.py). Only plain, serializable
        # values (strings, numbers) go into user_controls/state.
        session = _get_session(request.session_id)
        from src.tools.retriever_tool import RetrieverTool
        RetrieverTool.set_active_retriever(session.get("vector_retriever"))

        user_controls = dict(request.user_controls or {})
        user_controls["_has_vector_retriever"] = session.get("vector_retriever") is not None

        # Conversational memory: pass session_id through as the
        # LangGraph thread_id, so the checkpointer (configured in
        # GraphBuilder.build_graph) loads this session's prior turns
        # before running this query, and persists this turn for next
        # time. Different session_ids get fully isolated histories.
        response, tool_name = state._graph_builder.execute(
            state._compiled_graph,
            request.question,
            user_controls,
            thread_id = request.session_id,
        )

        logging.info("API: Query processed. Tool used: %s", tool_name or "None")

        return QueryResponse(
            answer = response if response else "No response generated.",
            tool_used = tool_name if tool_name else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("API: Query endpoint failed.")
        raise HTTPException(status_code = 500, detail = f"Query processing failed: {str(e)}")

