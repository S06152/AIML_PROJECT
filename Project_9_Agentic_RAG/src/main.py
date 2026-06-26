import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from src.ui.streamlit_app import StreamlitApp
from src.graph.workflow_graph import GraphBuilder
import warnings
warnings.filterwarnings("ignore")

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

            self._graph = GraphBuilder()

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
                    self._app.run_ingestion_pipeline(uploaded_files, self._user_input)

                    st.session_state["_files_signature"] = self._get_files_signature(uploaded_files)

            # Build Agent Workflow Graph
            graph = self._graph.build_graph()

            # Display workflow graph
            self._display_graph(graph)

            # Process user query
            if self._user_query:
                self._run_agent_workflow(self._user_query.strip(), graph)

        except Exception as e:
            logging.exception("Error while executing the Agentic RAG workflow.")
            raise CustomException(e, sys)

    def _display_graph(self, graph) -> None:
        """
        Display workflow graph in sidebar.
        """
        try:
            with st.sidebar:
                try:
                    graph_image = graph.get_graph().draw_mermaid_png()
                    st.image(graph_image, caption = "Agentic RAG Workflow", use_container_width = True )

                except Exception:
                    mermaid_text = graph.get_graph().draw_mermaid()
                    st.code(mermaid_text, language = "mermaid")

        except Exception as e:
            logging.warning("Unable to render workflow graph. Error: %s", str(e))

    def _run_agent_workflow(self, user_query: str, graph) -> None:
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
        tool_name = None

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching & generating answer..."):
                try:
                    logging.info("Executing Agentic RAG workflow for query: '%s'", user_query)

                    response, tool_name = self._graph.execute(graph, user_query)

                    if tool_name:
                        formatted_response = (
                            f"{response}\n"
                            f"[🛠️ Tool_Used : {tool_name}]"
                        )
                    else:
                        formatted_response = response
    
                    st.markdown(formatted_response)

                    logging.info("Agentic RAG response generated successfully. Tool invoked: %s", tool_name or "None")

                except Exception as e:
                    logging.exception("Agentic RAG workflow execution failed for query: '%s'", user_query)
                    raise CustomException(e, sys)

        if response:
            history_content = response
            
            if tool_name:
                history_content = (
                    f"{response}\n"
                    f"[🛠️ Tool_Used : {tool_name}]"
                )

            st.session_state["messages"].append({"role" : "assistant", "content" : history_content})