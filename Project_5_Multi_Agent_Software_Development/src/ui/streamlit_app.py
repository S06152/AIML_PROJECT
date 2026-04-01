# Standard Library Imports
import sys
import os
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from typing import Optional
from src.config.settings import Config
from src.ingestion.pdf_loader import PDFLoader
from src.chunking.chunk import ChunkingStrategy
from src.embedding.embedding import EmbeddingManager
from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.chain.qa_chain import QAChain
import warnings
warnings.filterwarnings("ignore")

class StreamlitApp:
    """
    Main Streamlit UI controller for the AUTOSAR SWS Multi-Agent Dev System.

    Responsibilities:
        - Load and display sidebar configuration controls.
        - Accept AUTOSAR SWS PDF uploads and run the RAG ingestion pipeline.
        - Provide a Q&A chat interface for asking questions about the SWS doc.
        - Accept user software development requests and run the 5-agent workflow.
        - Display agent outputs in a clean tabbed interface.

    Session State Keys Used:
        - ``vector_retriever`` : Configured LangChain retriever (after PDF upload).
        - ``qa_chain``         : Initialized QAChain instance.
        - ``chat_history``     : List of (question, answer) tuples.
        - ``workflow_result``  : Final DevTeamState from the agent pipeline.
    """

    def __init__(self):
        """
        Initialize StreamlitApp: parse config, prepare state.

        Raises:
            Raises Exception: If config loading fails.
        """
        try:
            logging.info("Initializing StreamlitApp.")
            self._config = Config()
            self._user_control: dict = {}

            logging.info("StreamlitApp initialized successfully.")

        except Exception as e:
            logging.exception("Error initializing StreamlitApp.")
            raise CustomException(e, sys)

    # MAIN UI LOADER
    def load_streamlit_ui(self):
        """
        Render the Streamlit page and sidebar configuration, return user controls.

        Returns:
            dict: User-selected configuration values:
                  GROQ_API_KEY, LLM_Model_Name, TEMPERATURE, TOKEN,
                  CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, EMBEDDING_MODEL.

        Raises:
            AutosarMASException: If rendering fails.
        """
        try:
            logging.info("Loading Streamlit UI.")

            page_title = "🚗 " + self._config.get_page_title()
            st.set_page_config(page_title = page_title, layout = "wide")
            st.header(page_title)

            # Load API keys from environment
            groq_api_key = st.secrets.get("GROQ_API_KEY")
            
            if not groq_api_key:
                st.warning("⚠️ GROQ_API_KEY not found. Enter your groq API key in Streamlit secrets to enable LLM funtionality")

            # Sidebar Configuration
            with st.sidebar:
                st.subheader("⚙️ Configuration")

                # Display masked status indicators (read-only)
                st.text_input("🔑 Groq API Key:", value = "✅ Configured" if groq_api_key else "❌ Not Set", disabled = True, key = "GROQ_API_KEY")

                selected_llm = st.selectbox("🧠 Select LLM Model", self._config.get_groq_model_options())
                self._user_control["LLM_MODEL"] = selected_llm

                temp_options = self._config.get_temperature()
                selected_temperature = st.slider("🔥 Temperature:", min_value = temp_options[0], max_value = temp_options[-1], value = temp_options[1])
                self._user_control["TEMPERATURE"] = selected_temperature

                token_options = self._config.get_token()
                selected_token = st.slider("📏 Max Tokens:", min_value = token_options[0], max_value = token_options[-1], value = token_options[1])
                self._user_control["TOKEN"] = selected_token
            
            # Store remaining config values for downstream use
            self._user_control["GROQ_API_KEY"] = groq_api_key
            self._user_control["CHUNK_SIZE"] = self._config.get_chunk_size()
            self._user_control["CHUNK_OVERLAP"] = self._config.get_chunk_overlap()
            self._user_control["TOP_K"] = self._config.get_top_k()
            self._user_control["EMBEDING_MODELS"] = self._config.get_embedding_model()

            logging.info("Streamlit UI rendered successfully.")
            return self._user_control

        except Exception as e:
            logging.exception("Fatal error while rendering Streamlit UI.")
            raise CustomException(e, sys)
    
    # -----------------------------------------------------------------------
    # RAG Ingestion Pipeline
    # -----------------------------------------------------------------------
    def run_ingestion_pipeline(self, uploaded_file, user_controls: dict) -> Optional[str]:
        """
        Run the full RAG ingestion pipeline on an uploaded AUTOSAR SWS PDF.

        Steps:
            1. PDFLoader    — extract per-page Documents.
            2. Chunking     — split into overlapping chunks.
            3. Embedding    — vectorize with HuggingFace model.
            4. ChromaStore  — index vectors in Chroma.
            5. Retriever    — configure top-K retriever.
            6. QAChain      — build and cache the RAG chain.

        Args:
            uploaded_file: Streamlit UploadedFile object (AUTOSAR SWS PDF).
            user_controls (dict): UI configuration dict.

        Returns:
            Optional[str]: Success message, or None on failure.
        """
        try:
            logging.info("Starting AUTOSAR SWS ingestion pipeline.")

            # Loading AUTOSAR SWS PDF
            loader = PDFLoader(uploaded_file)
            documents = loader.load_documents()
            logging.info(f"✅ PDF loaded: {len(documents)} pages extracted.")

            # ✂️ Chunking document
            chunker = ChunkingStrategy(chunk_size = user_controls["CHUNK_SIZE"], chunk_overlap = user_controls["CHUNK_OVERLAP"])
            chunks = chunker.split_documents_into_chunks(documents)
            logging.info(f"✅ Chunked into {len(chunks)} segments.")

            # 🔢 Generating embeddings (this may take ~30 seconds)
            embedding_mgr = EmbeddingManager(user_controls["EMBEDING_MODELS"])
            embeddings = embedding_mgr.create_embeddings()
            logging.info("✅ Embeddings ready.")

            # 🗄️ Indexing into ChromaDB
            vector_store_mgr = ChromaVectorStore(chunks, embeddings)
            vector_db = vector_store_mgr.create_vectorstore()
            logging.info("✅ ChromaDB index built.")

            # 🔍 Configuring retriever
            retriever_mgr = Retriever(vector_db, top_k = user_controls["TOP_K"])
            vector_retriever = retriever_mgr.get_retriever()

            qa_chain = QAChain(
                    retriever = vector_retriever,
                    groq_api_key = user_controls["GROQ_API_KEY"],
                    model_name = user_controls["LLM_MODEL"],
                    temperature = user_controls["TEMPERATURE"],
                    max_tokens = user_controls["TOKEN"]
                )

            # Store in session state for reuse across interactions
            st.session_state["vector_retriever"] = vector_retriever
            st.session_state["qa_chain"] = qa_chain
            st.session_state["chat_history"] = []

            logging.info("AUTOSAR SWS ingestion pipeline completed successfully.")

        except Exception as e:
            logging.exception("Ingestion pipeline failed.")
            raise CustomException(e, sys)
    

    # -----------------------------------------------------------------------
    # AUTOSAR Context Retrieval
    # -----------------------------------------------------------------------
    def retrieve_autosar_context(self, user_request: str) -> str:
        """
        Retrieve the most relevant AUTOSAR SWS chunks for a user request.

        This context is injected into the ProductManagerAgent's prompt so that
        all downstream agents are grounded in the actual SWS specification.

        Args:
            user_request (str): The user's software development request.

        Returns:
            str: Concatenated AUTOSAR SWS chunks (top-K), or empty string
                 if no document has been uploaded.
        """
        retriever = st.session_state.get("vector_retriever")
        if retriever is None:
            logging.info("No AUTOSAR SWS document loaded. Context will be empty.")
            return ""

        try:
            logging.info("Retrieving AUTOSAR SWS context for user request.")
            docs = retriever.invoke(user_request)
            context: str = "\n\n".join(doc.page_content for doc in docs)
            logging.info("Retrieved %d AUTOSAR SWS chunks.", len(docs))
            return context
        
        except Exception as e:
            logging.exception("Context retrieval failed")
            raise CustomException(e, sys)