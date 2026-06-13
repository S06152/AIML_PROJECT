import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from src.ui.streamlit_app import StreamlitApp
from src.graph.workflow_graph import GraphBuilder
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
            st.session_state["user_controls"] = self._user_input
            self._graph = GraphBuilder()

            self._user_query = st.chat_input("💬 Enter your question here...")
            
            logging.info("Orchestrator initialized. User controls loaded successfully.")

        except Exception as e:
            logging.exception("Failed to initialize MultiModalRAG orchestrator.")
            raise CustomException(e, sys)

    def _get_files_signature(self, uploaded_files):
        """
        Generate unique signature for uploaded files.
        Used to detect file changes for auto re-indexing.
        """
        if not uploaded_files:
            return None
        
        return tuple(sorted((f.name, f.size) for f in uploaded_files))
    
    def _needs_reindexing(self, uploaded_files):
        """
        Check whether documents changed.
        """
        current_sig = self._get_files_signature(uploaded_files)

        if current_sig is None:
            return False

        prev_sig = st.session_state.get("_files_signature")

        needs_update = (current_sig != prev_sig)

        if needs_update:
            logging.info("Reindexing triggered due to file or DB change.")

        return needs_update
      
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
            # Chat History
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
         
            # Phase 1: PDF Upload + RAG Ingestion
            uploaded_files = st.sidebar.file_uploader("📂 Upload PDF(s)", type = ["pdf"], accept_multiple_files = True, help = "Select one or more PDFs.")
            index_clicked = st.sidebar.button("⚡ Index PDF(s)", disabled = (not uploaded_files), use_container_width = True, type = "primary")
            
            if uploaded_files and index_clicked:
                if self._needs_reindexing(uploaded_files):
                    self._app.run_ingestion_pipeline(uploaded_files, self._user_input)
                    st.session_state["_files_signature"] = (self._get_files_signature(uploaded_files))

            # ---------------------------------------------------------------
            # Phase 2: User Query + RAG Response
            # ---------------------------------------------------------------
            graph = self._graph.build_graph()
            if self._user_query:
                self._run_agent_workflow(self._user_query.strip(), graph)

        except Exception as e:
            logging.exception("Error during RAG pipeline execution.")
            raise CustomException(e, sys)

    def _run_agent_workflow(self, user_query: str, graph) -> None:
        """
        Retrieve relevant context and generate a response for the user query.

        Args:
            user_query (str): Validated, non-empty user query string.

        Raises:
            CustomException: If context retrieval or response generation fails.
        """

        st.session_state["messages"].append({"role": "user", "content": user_query})

        # Display the user message immediately before generating response
        with st.chat_message("user"):
            st.markdown(user_query)

        # Initialize response before the block to avoid UnboundLocalError
        response = None
        tool_name = None

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching & generating answer..."):
                try:
                    logging.info("Starting RAG workflow for user query: '%s'", user_query)

                    # Step 1: Execute graph and get response + tool name
                    response, tool_name = self._graph.execute(graph, user_query)

                    # Display which tool was invoked
                    if tool_name:
                        st.info(f"🛠️ **Tool Called:** `{tool_name}`")
                    else:
                        st.info("💡 **Tool Called:** `None` (Direct LLM response)")

                    st.markdown(response)
                    
                    logging.info("Response generated successfully. Tool used: %s", tool_name or "None")

                except Exception as e:
                    logging.exception("RAG workflow failed for query: '%s'", user_query)
                    raise CustomException(e,sys)
                
        if response is not None:
            # Include tool name in the stored message for chat history
            tool_label = f"🛠️ **Tool Called:** `{tool_name}`\n\n" if tool_name else "💡 **Tool Called:** `None` (Direct LLM response)\n\n"
            st.session_state["messages"].append({"role": "assistant", "content": tool_label + response})
                
