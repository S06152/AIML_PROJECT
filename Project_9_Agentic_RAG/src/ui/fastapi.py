import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from fastapi import FastAPI, UploadFile, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import streamlit as st
from src.ingestion.pdf_loader import PDFLoader
from src.embedding.embedding import EmbeddingManager
from src.vectorstore.chroma_store import ChromaVectorStore
from src.tools.retriever_tool import RetrieverTool
from src.graph.workflow_graph import GraphBuilder
import warnings
warnings.filterwarnings("ignore")

class QueryRequest(BaseModel):
    question: str
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

class FastAPIState:
    def __init__(self):
        self._graph_builder = GraphBuilder()
        # Compile graph once at startup
        self._compiled_graph = self._graph_builder.build_graph()
        logging.info("Graph compiled once at FastAPI startup.")
        # Cache mermaid text for frontend rendering
        try:
            self._graph_mermaid = self._compiled_graph.get_graph().draw_mermaid()
        except Exception:
            self._graph_mermaid = None

app = FastAPI()

state = FastAPIState()

# Endpoints
@app.get("/health", response_model = HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status = "healthy")

@app.get("/graph")
async def get_graph():
    """Return the compiled graph mermaid text for frontend rendering."""
    return {"graph_mermaid": state._graph_mermaid}

@app.post("/upload", response_model = UploadResponse)
async def upload_documents(files: List[UploadFile], user_controls: dict):
    try:
        all_documents = []
        processed_files = []

        # Step 1: Load PDFs
        for file in files:
            try:
                loader = PDFLoader(file, user_controls)
                documents = loader.load_documents()
                all_documents.extend(documents)
                processed_files.append(file.filename)
                logging.info("PDF loaded successfully | File=%s | Pages=%s", file.name, len(documents))
            except Exception as e:
                logging.error(f"Failed to load {file.name}: {e}")
                st.error(f"❌ Failed to load {file.name}")
                continue
        
        if not all_documents:
            raise HTTPException(status_code = 400, details = "No valid PDF documnets could be processed")
        
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

        # Step 5: Store Retriever
        st.session_state["vector_retriever"] = vector_retriever
        logging.info("Retriever stored in session state.")

        logging.info("Document ingestion pipeline completed successfully.")

        return UploadResponse(
            message = "Documents indexed successfully.",
            pages_extracted = len(all_documents),
            files_processed = processed_files
        )
    except Exception as e:
        logging.exception("Document ingestion pipeline failed.")
        raise HTTPException(status_code = 500, details = f"Document processing failed: {str(e)}")

@app.post("/query", response_model = QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        logging.info("API: Query received: '%s'", request.question)

        if not request.question.strip():
            raise HTTPException(status_code = 400, detail = "Question cannot be empty")
        
        # Invoke the pre-compiled graph for each query
        response, tool_name = state._graph_builder.execute(
            state._compiled_graph, request.question, request.user_controls
        )

        logging.info("API: Query processed. Tool used: %s", tool_name or "None")

        return QueryResponse(
            answer = response if response else "No response generated.",
            tool_used = tool_name if tool_name else None
        )
    
    except Exception as e:
        logging.exception("API: Query endpoint failed.")
        raise HTTPException(status_code = 500, detail = f"Query processing failed: {str(e)}")

