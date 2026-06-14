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

            # Display the Agentic Workflow Graph in sidebar
            self._display_graph(graph)

            if self._user_query:
                self._run_agent_workflow(self._user_query.strip(), graph)

        except Exception as e:
            logging.exception("Error during RAG pipeline execution.")
            raise CustomException(e, sys)

    def _build_citation_footer(self, tool_name: str, tool_metadata: dict) -> str:
        """
        Build a Markdown citation footer to display below the LLM response.

        Rules:
          - vector_db_retriever  → show document name(s), page number(s), and tool name.
          - any other tool       → show source URL(s) and tool name.
          - no tool called       → return empty string (no footer).

        Args:
            tool_name     (str):  Name of the tool invoked (empty string if none).
            tool_metadata (dict): Metadata extracted by GraphBuilder._extract_tool_metadata.

        Returns:
            str: Markdown-formatted footer, or empty string if nothing to show.
        """
        if not tool_name:
            return ""

        lines = ["---", "**📎 Sources**"]

        if tool_name == "vector_db_retriever":
            sources = tool_metadata.get("sources", [])
            if sources:
                for s in sources:
                    lines.append(f"- 📄 **{s['doc']}** — Page {s['page']}")
            else:
                lines.append("- *(Retrieved from indexed documents — page details unavailable)*")

        else:
            urls = tool_metadata.get("urls", [])
            if urls:
                for url in urls:
                    lines.append(f"- 🔗 [{url}]({url})")
            else:
                lines.append("- *(Web source details unavailable)*")

        lines.append(f"\n🛠️ **Tool Used:** `{tool_name}`")
        return "\n".join(lines)

    def _display_graph(self, graph) -> None:
        """
        Display the Agentic RAG workflow graph in the Streamlit sidebar.

        Uses LangGraph's built-in Mermaid diagram rendering to produce
        a PNG image of the compiled state graph.

        Args:
            graph: Compiled LangGraph StateGraph instance.
        """
        try:
            with st.sidebar:
                st.subheader("🔀 Agent Workflow Graph")
                
                # Method 1: Try to render as PNG image (requires grandalf package)
                try:
                    graph_image = graph.get_graph().draw_mermaid_png()
                    st.image(graph_image, caption="Agentic RAG Workflow", use_container_width=True)
                except Exception:
                    # Method 2: Fallback to Mermaid text diagram
                    mermaid_text = graph.get_graph().draw_mermaid()
                    st.code(mermaid_text, language="mermaid")

        except Exception as e:
            logging.warning(f"Could not display workflow graph: {e}")

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
        tool_metadata = {}

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching & generating answer..."):
                try:
                    logging.info("Starting RAG workflow for user query: '%s'", user_query)

                    # Step 1: Execute graph and get response + tool name + source metadata
                    response, tool_name, tool_metadata = self._graph.execute(graph, user_query)

                    # Display which tool was invoked
                    if tool_name:
                        st.info(f"🛠️ **Tool Called:** `{tool_name}`")
                    else:
                        st.info("💡 **Tool Called:** `None` (Direct LLM response)")

                    st.markdown(response)

                    # Render citation footer
                    footer_md = self._build_citation_footer(tool_name, tool_metadata)
                    if footer_md:
                        st.markdown(footer_md)
                    
                    logging.info("Response generated successfully. Tool used: %s", tool_name or "None")

                except Exception as e:
                    logging.exception("RAG workflow failed for query: '%s'", user_query)
                    raise CustomException(e,sys)
                
        if response is not None:
            # Include tool name + footer in the stored message for chat history
            tool_label = f"🛠️ **Tool Called:** `{tool_name}`\n\n" if tool_name else "💡 **Tool Called:** `None` (Direct LLM response)\n\n"
            footer_md = self._build_citation_footer(tool_name, tool_metadata)
            history_content = tool_label + response
            if footer_md:
                history_content += "\n\n" + footer_md
            st.session_state["messages"].append({"role": "assistant", "content": history_content})
                
