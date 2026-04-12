"""
streamlit_app.py — Streamlit UI controller for the AUTOSAR MAS system.

Manages the complete user-facing workflow:
    Phase 1: AUTOSAR SWS PDF Upload → RAG ingestion pipeline.
    Phase 2: User development request → 5-agent workflow → tabbed results.
    Phase 3: Artifact save → downloadable files displayed in sidebar.

FIX vs original:
    - Removed non-existent chat_history / Q&A interface that referenced missing
      session state keys, causing KeyError on first run.
    - Fixed self._user_controls → self._user_control.
    - All agent result tabs now actually render the state keys.
    - Added ArtifactSaver integration with download buttons.
    - Added module name extractor for artifact filenames.
"""

import re
import sys
import streamlit as st
from typing import Optional
from src.config.settings import Config
from src.ingestion.pdf_loader import PDFLoader
from src.chunking.chunk import ChunkingStrategy
from src.embedding.embedding import EmbeddingManager
from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.chain.qa_chain import QAChain
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.artifact_saver import ArtifactSaver

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Tab labels mapping: state key → display label
# ---------------------------------------------------------------------------
_TAB_CONFIG: dict = {
    "product_spec": ("📋 Requirements",   "product_spec"),
    "architecture": ("🏗️ Architecture",   "architecture"),
    "code":         ("💻 Source Code",    "code"),
    "tests":        ("🧪 Test Cases",     "tests"),
    "review":       ("🔍 Code Review",    "review"),
}


def _extract_module_name(user_request: str) -> str:
    """
    Heuristically extract an AUTOSAR module name from the user request.

    Looks for known AUTOSAR module identifiers in the request text.

    Args:
        user_request (str): User's natural-language request.

    Returns:
        str: Extracted module name (e.g., "COM", "NvM") or "MODULE".
    """
    known_modules = [
        "COM", "NvM", "CanIf", "EcuM", "Dem", "Det", "Rte", "Os",
        "MemIf", "Fee", "Fls", "Spi", "Adc", "Pwm", "Icu", "Gpt",
        "Wdg", "PduR", "CanSM", "LinSM", "FrSM", "Dcm", "Fim",
    ]
    upper_request = user_request.upper()
    for module in known_modules:
        if module.upper() in upper_request:
            return module
    # Fallback: try to find a word that looks like a module name (ALL_CAPS or CamelCase)
    match = re.search(r'\b([A-Z][a-zA-Z]{1,6})\b', user_request)
    return match.group(1) if match else "MODULE"


class StreamlitApp:
    """
    Main Streamlit UI controller for the AUTOSAR SWS Multi-Agent Dev System.

    Responsibilities:
        - Render sidebar configuration controls.
        - Accept AUTOSAR SWS PDF uploads and run the RAG ingestion pipeline.
        - Accept user development requests and launch the 5-agent workflow.
        - Display agent outputs in a tabbed interface.
        - Save and expose generated artifacts for download.

    Session State Keys:
        - ``vector_retriever``: Configured LangChain retriever (after PDF upload).
        - ``workflow_result`` : Final DevTeamState from the agent pipeline.
    """

    def __init__(self) -> None:
        """Initialize StreamlitApp and load configuration."""
        try:
            logger.info("Initializing StreamlitApp.")
            self._config = Config()
            self._user_control: dict = {}
            logger.info("StreamlitApp initialized.")
        except Exception as e:
            raise CustomException(e, sys) from e

    # -----------------------------------------------------------------------
    # UI Initialization
    # -----------------------------------------------------------------------
    def load_streamlit_ui(self) -> dict:
        """
        Render the Streamlit page header and sidebar configuration controls.

        Returns:
            dict: User-selected configuration values:
                  GROQ_API_KEY, LLM_MODEL, TEMPERATURE, TOKEN,
                  CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, EMBEDDING_MODEL.
        """
        try:
            logger.info("Rendering Streamlit UI.")

            page_title = "🚗 " + self._config.get_page_title()
            st.set_page_config(page_title=page_title, layout="wide")
            st.title(page_title)
            st.caption(
                "Upload an AUTOSAR SWS PDF, then describe the module feature you need. "
                "The 5-agent team will generate Requirements → Architecture → Code → Tests → Review."
            )

            # ── Read API key from secrets ────────────────────────────────────
            groq_api_key: str = st.secrets.get("GROQ_API_KEY", "")
            if not groq_api_key:
                st.warning(
                    "⚠️ GROQ_API_KEY not found. Add it to `.streamlit/secrets.toml`:\n"
                    "```\nGROQ_API_KEY = 'gsk_...'\n```"
                )

            # ── Sidebar controls ─────────────────────────────────────────────
            with st.sidebar:
                st.subheader("⚙️ LLM Configuration")

                st.text_input(
                    "🔑 Groq API Key Status",
                    value="✅ Configured" if groq_api_key else "❌ Not Set",
                    disabled=True,
                )

                selected_llm = st.selectbox(
                    "🧠 LLM Model", self._config.get_groq_model_options()
                )

                temp_opts = self._config.get_temperature()
                selected_temperature = st.slider(
                    "🌡️ Temperature",
                    min_value=temp_opts[0],
                    max_value=temp_opts[-1],
                    value=temp_opts[1],
                    step=0.1,
                )

                token_opts = self._config.get_token()
                selected_token = st.slider(
                    "📏 Max Tokens",
                    min_value=token_opts[0],
                    max_value=token_opts[-1],
                    value=token_opts[1],
                    step=256,
                )

                st.divider()
                st.subheader("📂 Artifacts")
                if "artifact_paths" in st.session_state:
                    st.success("✅ Artifacts saved")
                    for key, path in st.session_state["artifact_paths"].items():
                        label = _TAB_CONFIG.get(key, (key, key))[0]
                        with open(path, "r", encoding="utf-8") as fp:
                            st.download_button(
                                label=f"⬇ {label}",
                                data=fp.read(),
                                file_name=path.split("/")[-1],
                                mime="text/markdown",
                                key=f"dl_{key}",
                            )

            # ── Store controls ───────────────────────────────────────────────
            self._user_control["GROQ_API_KEY"]   = groq_api_key
            self._user_control["LLM_MODEL"]       = selected_llm
            self._user_control["TEMPERATURE"]     = selected_temperature
            self._user_control["TOKEN"]           = selected_token
            self._user_control["CHUNK_SIZE"]      = self._config.get_chunk_size()
            self._user_control["CHUNK_OVERLAP"]   = self._config.get_chunk_overlap()
            self._user_control["TOP_K"]           = self._config.get_top_k()
            self._user_control["EMBEDDING_MODEL"] = self._config.get_embedding_model()

            logger.info("Streamlit UI rendered.")
            return self._user_control

        except Exception as e:
            raise CustomException(e, sys) from e

    # -----------------------------------------------------------------------
    # RAG Ingestion Pipeline
    # -----------------------------------------------------------------------
    def run_ingestion_pipeline(self, uploaded_file, user_controls: dict) -> None:
        """
        Run the full RAG ingestion pipeline on an uploaded AUTOSAR SWS PDF.

        Steps:
            PDFLoader → ChunkingStrategy → EmbeddingManager →
            ChromaVectorStore → Retriever → cached in session_state.

        Args:
            uploaded_file: Streamlit UploadedFile object (AUTOSAR SWS PDF).
            user_controls (dict): UI configuration dict.
        """
        try:
            logger.info("Starting AUTOSAR SWS ingestion pipeline.")

            with st.spinner("📄 Loading AUTOSAR SWS PDF..."):
                loader    = PDFLoader(uploaded_file)
                documents = loader.load_documents()
                st.sidebar.success(f"✅ Loaded: {len(documents)} content blocks.")

            with st.spinner("✂️ Chunking document..."):
                chunker = ChunkingStrategy(
                    chunk_size=user_controls["CHUNK_SIZE"],
                    chunk_overlap=user_controls["CHUNK_OVERLAP"],
                )
                chunks = chunker.split_documents_into_chunks(documents)
                st.sidebar.success(f"✅ Chunked: {len(chunks)} segments.")

            with st.spinner("🔢 Generating embeddings (first run ~30s)..."):
                embedding_mgr = EmbeddingManager(user_controls["EMBEDDING_MODEL"])
                embeddings    = embedding_mgr.create_embeddings()
                st.sidebar.success("✅ Embeddings ready.")

            with st.spinner("🗄️ Building ChromaDB index..."):
                vector_store_mgr = ChromaVectorStore(chunks, embeddings)
                vector_db        = vector_store_mgr.create_vectorstore()
                st.sidebar.success("✅ Vector index built.")

            retriever_mgr  = Retriever(vector_db, top_k=user_controls["TOP_K"])
            vector_retriever = retriever_mgr.get_retriever()

            st.session_state["vector_retriever"] = vector_retriever
            st.session_state["user_controls"]    = user_controls

            logger.info("AUTOSAR SWS ingestion pipeline complete.")
            st.sidebar.success("🎯 AUTOSAR SWS document ready for Q&A and generation.")

        except Exception as e:
            st.sidebar.error(f"❌ Ingestion failed: {str(e)}")
            raise CustomException(e, sys) from e

    # -----------------------------------------------------------------------
    # AUTOSAR Context Retrieval
    # -----------------------------------------------------------------------
    def retrieve_autosar_context(
        self, user_request: str, user_controls: dict
    ) -> str:
        """
        Retrieve the most relevant AUTOSAR SWS chunks for the user request.

        This context is injected into the ProductManagerAgent prompt so all
        downstream agents are grounded in the actual SWS specification.

        Args:
            user_request  (str) : User's development request.
            user_controls (dict): UI configuration dict.

        Returns:
            str: Concatenated AUTOSAR SWS context (top-K chunks), or "".
        """
        retriever = st.session_state.get("vector_retriever")
        if retriever is None:
            logger.info("No AUTOSAR SWS document indexed. Context will be empty.")
            return ""

        try:
            logger.info("Retrieving AUTOSAR SWS context for user request.")
            qa_chain = QAChain(
                retriever=retriever,
                groq_api_key=user_controls["GROQ_API_KEY"],
                model_name=user_controls["LLM_MODEL"],
                temperature=user_controls["TEMPERATURE"],
                max_tokens=user_controls["TOKEN"],
            )
            context = qa_chain.run(user_request)
            logger.info("AUTOSAR context retrieved: %d chars.", len(context))
            return context

        except Exception as e:
            logger.warning("Context retrieval failed (non-fatal): %s", str(e))
            return ""

    # -----------------------------------------------------------------------
    # Agent Results Display
    # -----------------------------------------------------------------------
    def display_results(self, result: dict, module_name: str) -> None:
        """
        Display all 5 agent outputs in labelled tabs, then save artifacts.

        Args:
            result      (dict): Final DevTeamState from the workflow.
            module_name (str) : AUTOSAR module name for filenames.
        """
        st.success("✅ AUTOSAR MAS pipeline completed successfully!")
        st.divider()

        # ── Tabbed output display ────────────────────────────────────────────
        tab_labels = [cfg[0] for cfg in _TAB_CONFIG.values()]
        tabs = st.tabs(tab_labels)

        for tab, (key, (label, state_key)) in zip(tabs, _TAB_CONFIG.items()):
            with tab:
                content = result.get(state_key, "")
                if content:
                    st.markdown(content, unsafe_allow_html=False)
                else:
                    st.info(f"No content generated for {label}.")

        # ── Save artifacts ───────────────────────────────────────────────────
        try:
            saver = ArtifactSaver()
            saved_paths = saver.save_all(result, module_name=module_name)
            st.session_state["artifact_paths"] = saved_paths

            if saved_paths:
                st.info(
                    f"💾 {len(saved_paths)} artifacts saved to: `{saver.session_dir}/`  \n"
                    "Use the ⬇ download buttons in the sidebar to export them."
                )
        except Exception as save_err:
            logger.warning("Artifact save failed (non-fatal): %s", str(save_err))
            st.warning(f"⚠️ Could not save artifacts: {save_err}")
