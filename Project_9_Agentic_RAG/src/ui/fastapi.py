import base64
import io
import json
import os
import threading
from dotenv import load_dotenv

load_dotenv()
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.agents.agents import Agent
from src.embedding.embedding import EmbeddingManager
from src.graph.workflow_graph import GraphBuilder
from src.ingestion.pdf_loader import PDFLoader
from src.tools.retriever_tool import RetrieverTool
from src.vectorstore.query_rewriter import QueryRewriter
from src.utils.logger import logging
from src.vectorstore.chroma_store import ChromaVectorStore
from src.vectorstore.Sparse_store import SparseVectorStore
from langchain_classic.retrievers import EnsembleRetriever
from src.vectorstore.reranker import DocumentReranker
from src.vectorstore.reranking_retriever import RerankingRetriever



class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    workspace_id: str = Field(min_length=1, max_length=128)
    user_controls: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    tool_used: Optional[str] = None
    rewritten_queries: List[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    message: str
    pages_extracted: int
    files_processed: List[str]
    workspace_id: str


class HealthResponse(BaseModel):
    status: str


class BackendServices:
    """Backend-owned runtime resources. No Streamlit state is used here.

    The in-memory registry is suitable for a single FastAPI process/local setup.
    For multi-worker or multi-replica production, replace it with persistent
    vector storage and resolve the retriever by workspace_id on each request.
    """

    def __init__(self):
        self._retrievers: Dict[str, object] = {}
        self._lock = threading.RLock()
        self._graph_image_b64 = None

    def set_retriever(self, workspace_id: str, retriever) -> None:
        with self._lock:
            self._retrievers[workspace_id] = retriever

    def get_retriever(self, workspace_id: str):
        with self._lock:
            return self._retrievers.get(workspace_id)

    def graph_image(self):
        if self._graph_image_b64 is None:
            try:
                graph = GraphBuilder(retriever=None).build_graph()
                png = graph.get_graph().draw_mermaid_png()
                self._graph_image_b64 = base64.b64encode(png).decode("utf-8")
            except Exception:
                logging.exception("Unable to generate workflow graph image")
        return self._graph_image_b64


app = FastAPI(title="Agentic RAG API", version="1.0.0")
services = BackendServices()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy")


@app.get("/graph")
async def get_graph():
    return {"graph_image": services.graph_image()}


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    workspace_id: str = Form(...),
    user_controls: str = Form("{}"),
):
    """Ingest PDFs for one workspace and register its retriever in backend memory."""
    try:
        controls = json.loads(user_controls)
        if not isinstance(controls, dict):
            raise HTTPException(status_code=400, detail="user_controls must be a JSON object")

        required = ("EMBEDDING_MODELS", "TOP_K", "CHUNK_SIZE", "CHUNK_OVERLAP", "CAPTION_MODEL")
        missing = [key for key in required if key not in controls]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing settings: {', '.join(missing)}")

        all_documents = []
        processed_files = []
        for upload in files:
            filename = upload.filename or "uploaded.pdf"
            content_type = upload.content_type or ""
            if content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
                logging.warning("Rejected non-PDF upload: %s", filename)
                continue

            raw = await upload.read()
            if not raw:
                continue

            # PDFLoader is kept independent of FastAPI and Streamlit. BytesIO
            # provides the synchronous file-like interface it expects.
            file_obj = io.BytesIO(raw)

            # Add attributes expected by PDFLoader
            file_obj.type = "application/pdf"
            file_obj.content_type = "application/pdf"
            file_obj.name = filename
            file_obj.filename = filename
            loader = PDFLoader(file_obj, controls)
            docs = loader.load_documents()
            all_documents.extend(docs)
            processed_files.append(filename)

        if not all_documents:
            raise HTTPException(status_code=400, detail="No valid PDF documents could be processed")

        embeddings = EmbeddingManager(controls["EMBEDDING_MODELS"]).create_embeddings()
        # Reranker model
        reranker_model = controls.get("RERANKER_MODEL")
         
        vector_db = ChromaVectorStore(all_documents, embeddings).create_vectorstore()
        dense_retriever = RetrieverTool(vector_db, top_k=int(controls["TOP_K"])).get_retriever()
        sparse_retriever = SparseVectorStore(all_documents, top_k=int(controls["TOP_K"])).create_Sparsevectorstore()
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.7, 0.3]
        )

        # Step 7: Initialize Cross-Encoder reranker
        reranker = DocumentReranker(
            model_name=reranker_model,
            top_k=int(controls["TOP_K"])
        )

        final_retriever = RerankingRetriever(
        base_retriever=ensemble_retriever,
        reranker=reranker,
        candidate_k=int(controls["TOP_K"])
    )
        services.set_retriever(workspace_id, final_retriever)

        logging.info("Indexed %d chunks for workspace %s", len(all_documents), workspace_id)
        return UploadResponse(
            message="Documents indexed successfully.",
            pages_extracted=len(all_documents),
            files_processed=processed_files,
            workspace_id=workspace_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception(
            "Document ingestion failed: %s",
            str(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {type(exc).__name__}: {str(exc)}"
        ) from exc
    
    

    finally:
        for upload in files:
            await upload.close()


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        if not os.getenv("GROQ_API_KEY"):
            raise HTTPException(status_code=503, detail="LLM provider is not configured on the server")

        rewriter = QueryRewriter(request.user_controls)
        enhanced_query = rewriter.rewrite(question)
        print("\n========== RAG PIPELINE DEBUG ==========")

        print("\n1. ORIGINAL USER QUERY:")
        print(question)

        print("\n2. ENHANCED QUERY:")
        print(enhanced_query)

        retriever = services.get_retriever(request.workspace_id)
        # Build a request/workspace-scoped graph so the retriever is injected
        # into the tool closure. No global Streamlit session is consulted.
        graph_builder = GraphBuilder(retriever=retriever)
        graph = graph_builder.build_graph()
        answer, tool_name = graph_builder.execute(
            graph=graph,
            query=enhanced_query,
            user_controls=request.user_controls
        )

        print("\n5. FINAL RESPONSE:")
        print(answer)
        
        return QueryResponse(
            answer=answer or "No response generated.",
            tool_used=tool_name or None,
            rewritten_queries=[enhanced_query]
        )
    
        

    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Query endpoint failed")
        raise HTTPException(status_code=500, detail="Query processing failed") from exc
