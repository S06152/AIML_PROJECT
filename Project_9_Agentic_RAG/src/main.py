import os
import uuid
import json
from dotenv import load_dotenv
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from src.ui.streamlit_app import StreamlitApp
import requests
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

FASTAPI_BASE_URL = "http://localhost:8000"

# LangSmith Configuration
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = "Agentic_RAG"

# Main Agentic RAG Application
class AGENTICRAG:
    """
    Top-level orchestrator for the Agentic RAG application.

    Responsibilities:
        - Initialize the Streamlit user interface.
        - Capture user configuration settings.
        - Manage document uploads and indexing.
        - Communicate with the FastAPI backend.
        - Execute Agentic RAG workflows using LangGraph.
        - Manage chat history and render responses.

    Usage:
        app = AGENTICRAG()
        app.run()
    """

    def __init__(self) -> None:
        """
        Initialize the Agentic RAG application.
        Initializes the Streamlit interface, user controls, workspace identifier, and chat input.

        Raises:
            CustomException: If initialization fails.
        """
        try:
            logging.info("Initializing Agentic RAG application.")

            # Initialize Streamlit UI
            self._app = StreamlitApp()
            self._user_input = self._app.load_streamlit_ui()

            # Store user configuration
            st.session_state["user_controls"] = self._user_input

            # Generate a unique workspace ID
            if "workspace_id" not in st.session_state:
                st.session_state["workspace_id"] = str(uuid.uuid4())

            # Capture user query
            self._user_query = st.chat_input("💬 Enter your question here...")

            logging.info("Agentic RAG application initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize Agentic RAG application.")
            raise CustomException(e, sys)

    # Document Signature Management
    def _get_files_signature(self, uploaded_files):
        """
        Generate a signature for uploaded documents.

        The signature helps identify whether uploaded files
        have changed and require re-indexing.

        Args:
            uploaded_files: Uploaded PDF files.

        Returns:
            tuple: Document signature or None.
        """
        if not uploaded_files:
            return None

        return tuple(sorted((f.name, f.size) for f in uploaded_files))

    def _needs_reindexing(self, uploaded_files) -> bool:
        """
        Determine whether uploaded documents require re-indexing.

        Returns:
            bool: True if document changes are detected,
                  otherwise False.
        """
        current_sig = self._get_files_signature(uploaded_files)

        if current_sig is None:
            return False

        prev_sig = st.session_state.get("_files_signature")

        needs_update = (current_sig != prev_sig)

        if needs_update:
            logging.info("Document change detected. Re-indexing required.")

        return needs_update

    # Main Application Workflow
    def run(self) -> None:
        """
        Execute the complete Agentic RAG workflow.

        Workflow:
            1. Display chat history.
            2. Handle document uploads.
            3. Index uploaded documents.
            4. Process user queries.
            5. Display generated responses.

        Raises:
            CustomException: If workflow execution fails.
        """
        try:
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            # Display previous chat messages
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Document Upload & Indexing
            uploaded_files = st.sidebar.file_uploader(
                "📂 Upload PDF(s)",
                type = ["pdf"],
                accept_multiple_files = True,
                help = "Select one or more PDFs."
            )

            index_clicked = st.sidebar.button(
                "⚡ Index PDF(s)",
                disabled = (not uploaded_files),
                use_container_width = True,
                type = "primary"
            )

            if uploaded_files and index_clicked:
                if self._needs_reindexing(uploaded_files):
                    success = self._run_ingestion_pipeline(uploaded_files, self._user_input)

                    # Update signature only after successful indexing
                    if success:
                        st.session_state["_files_signature"] = self._get_files_signature(uploaded_files)

            # Process user query
            if self._user_query:
                self._run_agent_workflow(self._user_query.strip())

        except Exception as e:
            logging.exception("Error while executing the Agentic RAG workflow.")
            raise CustomException(e, sys)

    # RAG Ingestion Pipeline
    def _run_ingestion_pipeline(self, uploaded_files, user_controls: dict) -> bool:
        """
        Execute the document ingestion pipeline.

        Steps:
            1. Prepare uploaded PDF files.
            2. Send documents to FastAPI.
            3. Generate embeddings in the backend.
            4. Create and configure vector stores.
            5. Configure the retriever.

        Args:
            uploaded_files: Uploaded PDF files.
            user_controls: User-selected configuration.

        Returns:
            bool: True if indexing succeeds, otherwise False.
        """

        try:
            with st.sidebar.spinner("🔎 Processing PDFs document...."):
                logging.info("Starting document ingestion pipeline.")
                logging.info("Number of uploaded PDF files: %d", len(uploaded_files))

                files_data = []
                for uploaded_file in uploaded_files:
                    uploaded_file.seek(0)
                    files_data.append((
                        "files",
                        (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                    ))

                ## Send documents to FastAPI backend
                response = requests.post(
                    f"{FASTAPI_BASE_URL}/upload",
                    files = files_data,
                    data = {
                        "workspace_id": st.session_state["workspace_id"],
                        "user_controls": json.dumps(user_controls)
                    },
                    timeout = 600
                )

                if response.status_code == 200:
                    result = response.json()
                    st.sidebar.success(
                        f"✅ {result['message']} "
                        f"({result['pages_extracted']} pages from "
                        f"{len(result['files_processed'])} file(s))"
                    )

                    logging.info("Documents uploaded and indexed successfully.")
                    return True

                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"❌ Upload failed: {error_detail}")
                    logging.error("Upload failed: %s", error_detail)
                    return False

        except requests.RequestException as e:
                st.error(f"❌ Backend connection error: {str(e)}")
                logging.exception("FastAPI backend connection failed during upload.")
                return False
        
        except Exception as e:
                st.error(f"❌ Upload error: {str(e)}")
                logging.exception("Document ingestion pipeline failed.")
                return False

     # Agentic RAG Query Workflow
    def _run_agent_workflow(self, user_query: str) -> None:
        """
        Execute the Agentic RAG workflow for a user query.

        Workflow:
            1. Capture user question.
            2. Send query to FastAPI backend.
            3. Execute the LangGraph workflow in the backend.
            4. Receive the generated answer.
            5. Display and store the response.

        Args:
            user_query (str): User question.

        Raises:
            CustomException: If workflow execution fails.
        """

        # Store user message in chat history
        st.session_state["messages"].append({"role" : "user", "content" : user_query})

        with st.chat_message("user"):
            st.markdown(user_query)

        response = None
        formatted_response = None

        # Get Response from FastAPI Backend
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching & generating answer..."):
                try:
                    logging.info("Executing Agentic RAG workflow for query: '%s'", user_query)
                    
                    user_controls = st.session_state.get("user_controls", {})
                    response = requests.post(
                        f"{FASTAPI_BASE_URL}/query",
                        json = {
                            "question": user_query,
                            "workspace_id": st.session_state["workspace_id"],
                            "user_controls": user_controls
                        },
                        timeout = 180
                    )

                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "No response generated.")
                        tool_used = result.get("tool_used")

                        # Format final response
                        if tool_used:
                            formatted_response = (
                                f"{answer}\n\n"
                                f"[🛠️ Tool_Used : {tool_used}]"
                            )
                        else:
                            formatted_response = answer

                        st.markdown(formatted_response)
                        logging.info("Final response received successfully.")
                    else:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"❌ Query failed: {error_detail}")
                        logging.error("Query failed with status code %d: %s", response.status_code, error_detail)

                except requests.RequestException as e:
                    logging.exception( "FastAPI backend connection failed during query.")
                    st.error(f"❌ Backend connection error: {str(e)}")

                except Exception as e:
                    logging.exception("Query to backend failed.")
                    st.error(f"❌ Error: {str(e)}")

        # Save Assistant Response in Chat History
        if formatted_response:
            st.session_state["messages"].append({"role" : "assistant", "content" : formatted_response})