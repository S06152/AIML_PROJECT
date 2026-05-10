import sys
import re
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.filename import slugify_filename
import streamlit as st
from src.ui.streamlit_app import StreamlitApp
from src.graph.workflow_graph import GraphBuilder
from src.report.pdf_generator import PDFGenerator
import warnings
warnings.filterwarnings("ignore")

MAX_REPORT_CHARS = 60000

class Load_Multi_Agent_Research_Report_Generator:
    """
    Main Orchestrator Class for Multi-Agent Research & Report Generator

    Responsibilities:
        - Initialize Streamlit UI
        - Capture user input
        - Trigger multi-agent workflow (LangGraph)
        - Store results in session state
        - Render final report + PDF download
    """

    def __init__(self)-> None:
        """
        Initialize application and UI components.
        """

        try:
            logging.info("ORCHESTRATOR INITIALIZATION START")

            # Initialize Streamlit UI handler
            logging.info("Initializing Streamlit UI components...")
            self._app = StreamlitApp()

            # Load UI elements (sidebar inputs, configs, etc.)
            self._user_input = self._app.load_streamlit_ui()

            logging.info("Streamlit UI loaded successfully with user configurations.")

            logging.info("ORCHESTRATOR INITIALIZATION COMPLETE")

        except Exception as e:
            logging.exception("ERROR during orchestrator initialization.")
            raise CustomException(e, sys)
        
    def run(self) -> None:
        """
        Entry point for user interaction.
        Handles chat input and triggers workflow.
        """
                
        try:
            logging.info("Waiting for user input...")

            # Chat input from Streamlit UI
            user_request = st.chat_input("Enter your research topic here...")
            
            if user_request:
                cleaned_request = user_request.strip()
                logging.info(f"User input received: {cleaned_request}")

                # Trigger agent workflow
                self._run_agent_workflow(cleaned_request)

        except Exception as e:
            logging.exception("ERROR while handling user input.")
            raise CustomException(e, sys)

    def _run_agent_workflow(self, user_request: str) -> None:
        """
        Execute multi-agent pipeline and render results.

        Pipeline:
            Orchestrator → Search → Extraction → Writer → Reviewer

        Args:
            user_request (str): User query/topic
        """

        try:
            logging.info("AGENT WORKFLOW START")

            # Initialize Workflow Graph
            with st.spinner("🤖 Initializing agent team..."):
                logging.info("Building workflow graph with user configurations...")
                workflow = GraphBuilder(self._user_input)

            logging.info("Workflow graph initialized successfully.")

            # Execute Multi-Agent Workflow
            with st.spinner("⚙️ Running multi-agent research pipeline..."):
                logging.info("Executing multi-agent workflow...")
                result = workflow.execute(user_request)
            
            logging.info("Multi-agent workflow execution completed successfully.")

            # Store Result in Session State
            st.session_state["workflow_result"] = result

            # Render Output
            self._render_output(result, user_request)

            logging.info("AGENT WORKFLOW END")

        except Exception as e:
            logging.exception("ERROR during multi-agent workflow execution.")
            raise CustomException(e,sys)
    
    def _render_output(self, result: dict, user_request: str = "") -> None:
        """
        Render final report and provide PDF download.

        Args:
            result (dict): Final workflow output
            user_request (str): Original user query (used to build a
                descriptive PDF filename).
        """
        try:
            logging.info("Rendering final output...")

            # Safely fetch report
            final_report = result.get("final_report") or result.get("draft_report")

            if not final_report:
                logging.warning("No report found in workflow result.")
                st.warning("⚠️ No report generated. Please try again.")
                return

            # Token / UI safety guard
            final_report = final_report[:MAX_REPORT_CHARS]

            logging.info("Report ready | Length = %d", len(final_report))

            # Display Report
            st.subheader("📄 Generated Research Report")
            st.markdown(final_report)

            # Generate PDF
            logging.info("Generating PDF...")
            pdf_buffer = PDFGenerator.generate_pdf(final_report)

            # Build a descriptive filename from the user query.
            # As a smart fallback, use the first H1 in the report
            # (the Writer agent always emits one).
            title_match = re.search(r"^\s*#\s+(.+)$", final_report, re.MULTILINE)
            fallback_title = title_match.group(1).strip() if title_match else ""
            pdf_filename = slugify_filename(user_request, fallback_title)
            logging.info("PDF filename = %s", pdf_filename)

            # Download Button
            st.download_button(
                label = "⬇️ Download Report as PDF",
                data = pdf_buffer,
                file_name = pdf_filename,
                mime = "application/pdf"
            )

            logging.info("PDF download ready.")

        except Exception as e:
            logging.exception("ERROR while rendering output.")
            raise CustomException(e, sys)
                
