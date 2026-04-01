import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from src.ui.streamlit_app import StreamlitApp
from src.graph.workflow_graph import DevTeamWorkflow
import warnings
warnings.filterwarnings("ignore")

class Load_Multi_Agent_Software_Development:
    """
    Top-level orchestrator for the AUTOSAR SWS Multi-Agent Dev System.

    Responsibilities:
        - Initialize the Streamlit UI and collect user configuration.
        - Provide the ``run()`` entry point called by app.py.
        - Coordinate the PDF upload → RAG → agent workflow sequence.

    Usage (from app.py):
        pipeline = Load_Multi_Agent_Software_Development()
        pipeline.run()
    """

    def __init__(self)-> None:
        """
        Initialize the orchestrator: build the Streamlit UI and collect controls.

        Raises:
            Raises Exception: If UI initialization fails.
        """
        try:
            logging.info("Initializing Load_Multi_Agent_Software_Development.")
            self._app = StreamlitApp()
            self._user_input = self._app.load_streamlit_ui()
            
            logging.info("Orchestrator initialized. User controls loaded.")

        except Exception as e:
            logging.exception("Orchestrator not initialized.")
            raise CustomException(e, sys)
        
    def run(self) -> None:
        """
        Execute the full application flow:

        Phase 1 — Document Ingestion (sidebar):
            User uploads AUTOSAR SWS PDF → RAG pipeline runs → retriever ready.

        Phase 2 — Development Request (main panel):
            User types a request → AUTOSAR context retrieved → 5 agents run →
            results displayed in tabs.

        Raises:
            Raises Exception: Propagated from sub-components on fatal error.
        """

        try:
            # ---------------------------------------------------------------
            # Phase 1: AUTOSAR SWS PDF Upload + RAG Ingestion
            # ---------------------------------------------------------------
            uploaded_files = st.sidebar.file_uploader("📂 Upload AUTOSAR SWS PDF", type = ["pdf"], accept_multiple_files = False, help = "Supported only: .pdf")
            
            if uploaded_files:
                st.sidebar.spinner("📄 Loading & processing documents...")
                self._app.run_ingestion_pipeline(uploaded_files, self._user_input)

            else:
                logging.info(
                    "⚠️ No AUTOSAR SWS document uploaded.\n\n"
                    "Agents will run without AUTOSAR specification context.\n"
                    "Upload a SWS PDF for grounded, compliant output."
                )

            # ---------------------------------------------------------------
            # Phase 2: Development Request + Agent Workflow
            # ---------------------------------------------------------------
            user_request = st.chat_input("Enter your question here...")
            if user_request:
                self._run_agent_workflow(user_request.strip())

        except Exception as e:
            logging.exception("Document not uploaded")
            raise CustomException(e, sys)

    def _run_agent_workflow(self, user_request: str) -> None:
        """
        Retrieve AUTOSAR context and execute the 5-agent pipeline.

        Args:
            user_request (str): Validated, non-empty user development request.
        """
        try:
            logging.info("Starting 5-agent pipeine for user request.")

                        # Retrieve AUTOSAR SWS context (grounding for all agents)
            with st.spinner("🔍 Retrieving AUTOSAR SWS context..."):
                autosar_context: str = self._app.retrieve_autosar_context(user_request)

            if autosar_context:
                st.info(
                    f"📄 AUTOSAR SWS context retrieved: "
                    f"{len(autosar_context)} chars from indexed document."
                )
            else:
                st.warning(
                    "⚠️ No AUTOSAR SWS document indexed. "
                    "Agents will generate output without specification grounding."
                )
            
            # Initialize DevTeamWorkflow with current user controls
            with st.spinner("🤖 Initializing agent team..."):
                workflow = DevTeamWorkflow(self._user_controls)

            with st.spinner("⚙️ Running 5-agent AUTOSAR development pipeline..."):
                # Execute workflow
                result = workflow.execute(user_request, autosar_context)
            
            logging.info("5-agent pipeline completed successfully.")

        except Exception as e:
            logging.exception("Agent workflow failed")
            raise CustomException(e,sys)
                
