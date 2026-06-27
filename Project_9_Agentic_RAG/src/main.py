import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from src.ui.streamlit_app import StreamlitApp
import requests
import warnings
warnings.filterwarnings("ignore")

FASTAPI_BASE_URL = "http://localhost:8000"

class AGENTICRAG:
    """
    Top-level orchestrator for the Agentic RAG application.

    Responsibilities:
        - Initialize the Streamlit user interface.
        - Capture and manage user configuration settings.
        - Coordinate document ingestion and indexing workflows.
        - Execute Agentic RAG query processing using LangGraph.
        - Manage chat interactions, tool invocation, and response rendering.

    Usage:
        app = AGENTICRAG()
        app.run()
    """

    def __init__(self) -> None:
        """
        Initialize the Agentic RAG application components.

        This includes:
            - Loading the Streamlit interface.
            - Capturing user configuration settings.
            - Initializing the workflow graph builder.
            - Preparing the chat input interface.

        Raises:
            CustomException: If initialization fails.
        """
        try:
            logging.info("Initializing Agentic RAG application.")

            self._app = StreamlitApp()
            self._user_input = self._app.load_streamlit_ui()

            st.session_state["user_controls"] = self._user_input

            self._user_query = st.chat_input("💬 Enter your question here...")

            logging.info("Agentic RAG application initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize Agentic RAG application.")
            raise CustomException(e, sys)

    def _get_files_signature(self, uploaded_files):
        """
        Generate a unique signature for uploaded documents.

        The signature is used to determine whether the uploaded
        documents have changed and require re-indexing.
        """
        if not uploaded_files:
            return None

        return tuple(sorted((f.name, f.size) for f in uploaded_files))

    def _needs_reindexing(self, uploaded_files):
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

    def run(self) -> None:
        """
        Execute the complete Agentic RAG workflow.

        Workflow:
            1. Display chat history.
            2. Handle document uploads and indexing.
            3. Build and display the workflow graph.
            4. Process user queries through the Agentic RAG pipeline.
            5. Render responses and tool usage

        Raises:
            CustomException: Propagated from underlying components.
        """
        try:
            # Chat History
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

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
                    self._run_ingestion_pipeline(uploaded_files, self._user_input)

                    st.session_state["_files_signature"] = self._get_files_signature(uploaded_files)

            # Process user query
            if self._user_query:
                self._run_agent_workflow(self._user_query.strip())

        except Exception as e:
            logging.exception("Error while executing the Agentic RAG workflow.")
            raise CustomException(e, sys)

    # RAG Ingestion Pipeline
    def _run_ingestion_pipeline(self, uploaded_files, user_controls: dict) -> None:
        """
        Execute the document ingestion pipeline.

        Steps:
            1. Load PDF documents.
            2. Generate embeddings.
            3. Create Chroma vector store.
            4. Configure retriever.
            5. Store retriever in session state.

        Args:
            uploaded_files: Uploaded PDF files.
            user_controls: User-selected configuration.
        """
        with st.sidebar.spinner("🔎 Processing PDFs document...."):
            logging.info("Starting document ingestion pipeline.")

            # Step 1: Load PDFs
            try:
                files_data = []
                for file in uploaded_files:
                    file.seek(0)
                    files_data.append(file)
                    
                # Send to FastAPI backend
                response = requests.post(
                    f"{FASTAPI_BASE_URL}/upload",
                    files = files_data,
                    data = user_controls
                )

                if response.status_codes == 200:
                    result = response.json()
                    st.success(
                        f"✅ {result['message']} "
                        f"({result['pages_extracted']} pages from "
                        f"{len(result['files_processed'])} file(s))"
                    )

                    logging.info("Documents uploaded and indexed successfully.")
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"❌ Upload failed: {error_detail}")
                    logging.error("Upload failed: %s", error_detail)

            except Exception as e:
                st.error(f"❌ Upload error: {str(e)}")
                logging.exception("Upload to backend failed.")

    def _run_agent_workflow(self, user_query: str) -> None:
        """
        Execute the Agentic RAG workflow for a user query.

        The workflow may:
            - Generate a direct LLM response.
            - Invoke a retriever tool.
            - Invoke external knowledge tools.
            - Return source attribution information.

        Args:
            user_query (str): User question submitted through the chat interface.
            graph: Compiled LangGraph workflow.

        Raises:
            CustomException: If workflow execution fails.
        """

        st.session_state["messages"].append({"role" : "user", "content" : user_query})

        with st.chat_message("user"):
            st.markdown(user_query)

        response = None
        formatted_response = None
        # Get response from backend
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching & generating answer..."):
                try:
                    logging.info("Executing Agentic RAG workflow for query: '%s'", user_query)
                    
                    response = requests.post(
                        f"{FASTAPI_BASE_URL}/query",
                        json = {"question" : user_query}
                    )

                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "No response generated.")
                        tool_used = result.get("tool_used")

                        if tool_used:
                            formatted_response = (
                                f"{answer}\n\n"
                                f"[🛠️ Tool_Used : {tool_used}]"
                            )
                        else:
                            formatted_response = answer

                        st.markdown(formatted_response)
                    else:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"❌ Query failed: {error_detail}")

                    logging.info("Agentic RAG response generated successfully.")

                except Exception as e:
                    logging.exception("Query to backend failed.")
                    st.error(f"❌ Error: {str(e)}")

        if formatted_response:
            history_content = formatted_response

            st.session_state["messages"].append({"role" : "assistant", "content" : history_content})