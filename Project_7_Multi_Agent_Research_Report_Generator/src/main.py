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

    MAX_REPORT_CHARS = 60000

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

            # Persistent history of all reports generated in this browser
            # session. Each entry is a dict:
            #   { "query": str, "report": str, "pdf_bytes": bytes,
            #     "filename": str }
            if "reports" not in st.session_state:
                st.session_state["reports"] = []

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

            # If a new query is submitted, compute and append the result
            # to the history BEFORE rendering, so it shows up in order.
            user_request = st.chat_input("Enter your research topic here...")

            if user_request:
                cleaned_request = user_request.strip()
                logging.info(f"User input received: {cleaned_request}")

                # Trigger agent workflow (appends to st.session_state.reports)
                self._compute_and_store(cleaned_request)

            # Always render the FULL history (previous + current).
            self._render_history()

        except Exception as e:
            logging.exception("ERROR while handling user input.")
            raise CustomException(e, sys)

    def _compute_and_store(self, user_request: str) -> None:
        """
        Execute multi-agent pipeline and APPEND result to session history.

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

            # Safely fetch report
            final_report = (result.get("final_report") or result.get("draft_report") or "")

            if not final_report.strip():
                logging.warning("No report found in workflow result.")
                st.warning("⚠️ No report generated for query: "f"'{user_request}'. Please try again.")
                return

            # Token / UI safety guard
            final_report = final_report[:self.MAX_REPORT_CHARS]

            # Build PDF bytes (must persist across reruns -> store bytes,
            # not BytesIO)
            logging.info("Generating PDF...")
            pdf_buffer = PDFGenerator.generate_pdf(final_report)
            pdf_bytes = pdf_buffer.getvalue()

            # Build a descriptive filename
            title_match = re.search(r"^\s*#\s+(.+)$", final_report, re.MULTILINE)

            fallback_title = (title_match.group(1).strip() if title_match else "")
            
            pdf_filename = slugify_filename(user_request, fallback_title)

            logging.info("PDF filename = %s", pdf_filename)

            # Append to history (newest at the end => keep original order)
            st.session_state["reports"].append(
                {
                    "query": user_request,
                    "report": final_report,
                    "pdf_bytes": pdf_bytes,
                    "filename": pdf_filename,
                }
            )

            logging.info("AGENT WORKFLOW END")

        except Exception as e:
            logging.exception("ERROR during multi-agent workflow execution.")
            raise CustomException(e, sys)

    def _render_history(self) -> None:
        """
        Render every report generated so far in this Streamlit session.

        Each report is shown inside an expander with its own download
        button (keyed uniquely so Streamlit doesn't complain).
        """

        try:
            reports = st.session_state.get("reports", [])

            if not reports:
                # Nothing to render yet; show a friendly hint once.
                st.info(
                    "� Enter a research topic in the chat box below to "
                    "generate your first report."
                )
                return

            # Toolbar: count + clear-history button
            top_left, top_right = st.columns([4, 1])

            with top_left:
                st.caption(
                    f"🗂 {len(reports)} report(s) in this session "
                    "(newest at the bottom)."
                )

            with top_right:
                if st.button("🗑 Clear history", use_container_width=True):
                    st.session_state["reports"] = []
                    st.rerun()

            # Render each report. Newest is the LAST one and is auto-expanded.
            last_idx = len(reports) - 1
            for idx, item in enumerate(reports):
                expanded = idx == last_idx
                header = f"📄 Query {idx + 1}: {item['query']}"

                with st.expander(header, expanded = expanded):
                    st.markdown(item["report"])
                    st.download_button(
                        label = "⬇️ Download Report as PDF",
                        data = item["pdf_bytes"],
                        file_name = item["filename"],
                        mime = "application/pdf",
                        key = f"download_pdf_{idx}",
                    )

        except Exception as e:
            logging.exception("ERROR while rendering history.")
            raise CustomException(e, sys)
                
