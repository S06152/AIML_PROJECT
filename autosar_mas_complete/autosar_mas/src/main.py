"""
main.py — Top-level orchestrator for the AUTOSAR MAS application.

Coordinates the three-phase application flow:
    Phase 1: PDF Upload → RAG ingestion (sidebar).
    Phase 2: User request → AUTOSAR context retrieval.
    Phase 3: 5-agent LangGraph workflow → tabbed result display → artifact save.

FIX vs original:
    - Fixed `self._user_controls` (undefined) → `self._user_control`.
    - Fixed workflow init: passes `self._user_input` (correct attribute name).
    - Added module_name extraction and passing through to workflow and saver.
    - Added proper error display in the Streamlit UI on failure.
"""

import sys
import streamlit as st
from src.ui.streamlit_app import StreamlitApp, _extract_module_name
from src.graph.workflow_graph import DevTeamWorkflow
from src.utils.logger import logger
from src.utils.exception import CustomException

import warnings
warnings.filterwarnings("ignore")


class AutosarMASOrchestrator:
    """
    Top-level orchestrator for the AUTOSAR SWS Multi-Agent Dev System.

    Responsibilities:
        - Initialize the Streamlit UI and collect user configuration.
        - Coordinate the PDF upload → RAG → 5-agent workflow sequence.
        - Delegate all rendering to StreamlitApp.

    Usage (from app.py):
        orchestrator = AutosarMASOrchestrator()
        orchestrator.run()
    """

    def __init__(self) -> None:
        """
        Initialize the orchestrator: build the Streamlit UI and collect controls.

        Raises:
            CustomException: If UI initialization fails.
        """
        try:
            logger.info("Initializing AutosarMASOrchestrator.")
            self._app        = StreamlitApp()
            # FIX: store as self._user_input (matches usage below)
            self._user_input = self._app.load_streamlit_ui()
            logger.info("Orchestrator initialized. UI controls loaded.")

        except Exception as e:
            raise CustomException(e, sys) from e

    def run(self) -> None:
        """
        Execute the full three-phase application flow.

        Phase 1 — PDF Upload (sidebar):
            User uploads an AUTOSAR SWS PDF → RAG ingestion pipeline runs.

        Phase 2 — Context Retrieval:
            User's request is used to retrieve the most relevant SWS chunks.

        Phase 3 — Agent Workflow:
            5-agent pipeline runs → results displayed in tabs → artifacts saved.
        """
        try:
            # ── Phase 1: AUTOSAR SWS PDF Upload + RAG Ingestion ─────────────
            uploaded_file = st.sidebar.file_uploader(
                "📂 Upload AUTOSAR SWS PDF",
                type=["pdf"],
                accept_multiple_files=False,
                help="Upload an AUTOSAR Classic Platform SWS specification PDF.",
            )

            if uploaded_file:
                # Re-run ingestion only when a new file is uploaded
                file_key = f"ingested_{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.get("last_ingested_key") != file_key:
                    self._app.run_ingestion_pipeline(uploaded_file, self._user_input)
                    st.session_state["last_ingested_key"] = file_key
            else:
                st.sidebar.info(
                    "📌 Upload an AUTOSAR SWS PDF to enable specification-grounded generation.\n\n"
                    "Without a PDF, agents will generate generic AUTOSAR output."
                )

            # ── Phase 2 & 3: Development Request + Agent Workflow ───────────
            user_request = st.chat_input(
                "Describe the AUTOSAR feature to implement "
                "(e.g., 'Implement COM module signal transmission')"
            )

            if user_request:
                self._run_agent_workflow(user_request.strip())

        except Exception as e:
            st.error(f"❌ Application error: {str(e)}")
            raise CustomException(e, sys) from e

    def _run_agent_workflow(self, user_request: str) -> None:
        """
        Run the full AUTOSAR context retrieval + 5-agent pipeline.

        Args:
            user_request (str): Validated, non-empty user development request.
        """
        try:
            logger.info("Starting AUTOSAR MAS pipeline for request: '%s'.", user_request[:80])

            # Display user's request in the chat-style UI
            with st.chat_message("user"):
                st.write(user_request)

            with st.chat_message("assistant"):
                # ── Step 1: Retrieve AUTOSAR SWS context ────────────────────
                with st.spinner("🔍 Retrieving AUTOSAR SWS context from indexed document..."):
                    autosar_context: str = self._app.retrieve_autosar_context(
                        user_request, self._user_input
                    )

                if autosar_context:
                    st.info(
                        f"📄 AUTOSAR SWS context retrieved: {len(autosar_context):,} chars "
                        f"from indexed specification."
                    )
                else:
                    st.warning(
                        "⚠️ No AUTOSAR SWS document indexed. "
                        "Agents will generate without specification grounding. "
                        "Upload a SWS PDF for best results."
                    )

                # ── Step 2: Extract module name ──────────────────────────────
                module_name: str = _extract_module_name(user_request)
                st.caption(f"🔧 Detected AUTOSAR module: **{module_name}**")

                # ── Step 3: Initialize 5-agent workflow ──────────────────────
                with st.spinner("🤖 Initializing 5-agent AUTOSAR development team..."):
                    # FIX: use self._user_input (not self._user_controls)
                    workflow = DevTeamWorkflow(self._user_input)

                # ── Step 4: Run workflow ─────────────────────────────────────
                agent_stages = [
                    "📋 ProductManager: Generating requirements...",
                    "🏗️ Architect: Designing system architecture...",
                    "💻 Developer: Writing AUTOSAR source code...",
                    "🧪 QA: Generating test cases...",
                    "🔍 Reviewer: Performing code review...",
                ]

                progress = st.progress(0, text="Starting agents...")
                for i, stage_text in enumerate(agent_stages):
                    progress.progress(
                        int((i / len(agent_stages)) * 90),
                        text=stage_text,
                    )

                with st.spinner("⚙️ Running AUTOSAR MAS pipeline (this may take 1–3 minutes)..."):
                    result = workflow.execute(
                        user_request=user_request,
                        autosar_context=autosar_context,
                        module_name=module_name,
                    )

                progress.progress(100, text="✅ Pipeline complete!")
                logger.info("5-agent pipeline completed.")

                # ── Step 5: Display results + save artifacts ─────────────────
                self._app.display_results(result, module_name=module_name)

        except Exception as e:
            st.error(f"❌ Agent workflow failed: {str(e)}")
            logger.error("Agent workflow error: %s", str(e))
            raise CustomException(e, sys) from e
