import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from src.ui.streamlit_app import StreamlitApp
import warnings
warnings.filterwarnings("ignore")

class MultiModalRAG:
    """
    Top-level orchestrator for the Multi-Modal RAG Pipeline.

    Responsibilities:
        - Initialize the Streamlit UI and collect user configuration.
        - Provide the ``run()`` entry point called by app.py.
        - Coordinate the PDF upload → ingestion → retrieval → response workflow.

    Usage (from app.py):
        pipeline = MultiModalRAG()
        pipeline.run()
    """

    def __init__(self)-> None:
        """
        Initialize the orchestrator: build the Streamlit UI and collect controls.

        Raises:
            CustomException: If UI initialization fails.
        """
        try:
            logging.info("Initializing MultiModalRAG orchestrator.")
            self._app = StreamlitApp()
            self._user_input = self._app.load_streamlit_ui()
            
            logging.info("Orchestrator initialized. User controls loaded successfully.")

        except Exception as e:
            logging.exception("Failed to initialize MultiModalRAG orchestrator.")
            raise CustomException(e, sys)
        
    def run(self) -> None:
        """
        Execute the full Multi-Modal RAG application flow.

        Phase 1 — Document Ingestion:
            User uploads PDF(s) via sidebar → RAG ingestion pipeline runs →
            retriever becomes ready for querying.

        Phase 2 — Query & Retrieval:
            User enters a query → relevant context retrieved from indexed
            documents → response generated and displayed.

        Raises:
            CustomException: Propagated from sub-components on fatal error.
        """

        try:
            # ---------------------------------------------------------------
            # Phase 1: PDF Upload + RAG Ingestion
            # ---------------------------------------------------------------
            uploaded_files = st.sidebar.file_uploader("📂 Upload PDF(s)", type = ["pdf"], accept_multiple_files = False, help = "Select one or more PDFs.")

            if uploaded_files:
                self._app.run_ingestion_pipeline(uploaded_files, self._user_input)

            else:
                st.warning(
                    "⚠️ No documents uploaded. "
                    "Please upload PDF(s) from the sidebar to enable context-aware responses."
                )

            # ---------------------------------------------------------------
            # Phase 2: User Query + RAG Response
            # ---------------------------------------------------------------
            user_query = st.chat_input("💬 Enter your question here...")
            if user_query:
                self._run_agent_workflow(user_query.strip())

        except Exception as e:
            logging.exception("Error during RAG pipeline execution.")
            raise CustomException(e, sys)

    def _run_agent_workflow(self, user_query: str) -> None:
        """
        Retrieve relevant context and generate a response for the user query.

        Args:
            user_query (str): Validated, non-empty user query string.

        Raises:
            CustomException: If context retrieval or response generation fails.
        """
        try:
            logging.info("Starting RAG workflow for user query: '%s'", user_query)

            # Step 1: Retrieve relevant context from indexed documents
            with st.spinner("🔍 Retrieving relevant context from documents..."):
                response: str = self._app.retrieve_context(user_query, self._user_input)

            # Display in chat format
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                st.markdown(response)

            # Persist chat history
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            st.session_state["chat_history"].append((user_query, response))

            logging.info("Response generated and displayed successfully.")

            logging.info("RAG workflow completed successfully.")

        except Exception as e:
            logging.exception("RAG workflow failed for query: '%s'", user_query)
            raise CustomException(e,sys)
                
