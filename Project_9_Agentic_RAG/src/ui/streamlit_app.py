# Standard Library Imports
import sys
import os
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from typing import Optional
from src.config.settings import Config
from src.ingestion.pdf_loader import PDFLoader
from src.embedding.embedding import EmbeddingManager
from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.chain.qa_chain import QAChain
import warnings
warnings.filterwarnings("ignore")

class StreamlitApp:
    """
    Main Streamlit UI controller for the Multi-Modal RAG Pipeline.

    Responsibilities:
        - Load and display sidebar configuration controls.
        - Accept PDF uploads and run the RAG ingestion pipeline.
        - Provide a Q&A chat interface for querying indexed documents.
        - Display retrieved context and generated responses.

    Session State Keys Used:
        - ``vector_retriever`` : Configured LangChain retriever (after PDF upload).
        - ``qa_chain``         : Initialized QAChain instance.
        - ``chat_history``     : List of (question, answer) tuples.
    """

    def __init__(self):
        """
        Initialize StreamlitApp: parse config, prepare state.

        Raises:
            CustomException: If config loading fails.
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
        Render the Streamlit page and sidebar configuration.

        Returns:
            dict: User-selected configuration values including:
                  GROQ_API_KEY, LLM_MODEL, TEMPERATURE, TOKEN,
                  CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, EMBEDDING_MODELS,
                  CAPTION_MODEL, COLLECTION, uploaded_files.

        Raises:
            CustomException: If rendering fails.
        """
        try:
            logging.info("Loading Streamlit UI.")

            page_title = "📚 " + self._config.get_page_title()
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
            self._user_control["EMBEDDING_MODELS"] = self._config.get_embedding_model()
            self._user_control["CAPTION_MODEL"] = self._config.get_caption_model()

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
        Run the full RAG ingestion pipeline on uploaded PDF(s).

        Steps:
            1. PDFLoader        — extract per-page Documents from all PDFs.
            2. ChunkingStrategy — split into overlapping chunks.
            3. EmbeddingManager — vectorize with the configured embedding model.
            4. ChromaVectorStore— index vectors in ChromaDB.
            5. Retriever        — configure top-K retriever.
            6. QAChain          — build and cache the RAG chain in session state.

        Args:
            uploaded_files (list): List of Streamlit UploadedFile objects.
            user_controls (dict): UI configuration dict.

        Raises:
            CustomException: If any step in the pipeline fails.
        """
        try:
            with st.sidebar.spinner("🔎 Processing PDFs document...."):
                logging.info("Starting RAG ingestion pipeline.")

                # Step 1: Load PDFs
                loader = PDFLoader(uploaded_file, user_controls)
                documents = loader.load_documents()
                logging.info(f"✅ PDF loaded: {len(documents)} pages extracted.")
                st.write("✅ PDF loaded successfully. Extracted {} pages.".format(len(documents)))
                
                # Step 2: Chunk documents
                #chunker = ChunkingStrategy(documents = documents, chunk_size = user_controls["CHUNK_SIZE"], chunk_overlap = user_controls["CHUNK_OVERLAP"])
                #chunks = chunker.split_documents()
                #logging.info("Chunking complete: %d chunks created.", len(chunks))
                #st.write(f"✅ Created {len(chunks)} chunk(s).")
                
                # Step 3: Generate embeddings
                embedding_mgr = EmbeddingManager(user_controls["EMBEDDING_MODELS"])
                embeddings = embedding_mgr.create_embeddings()
                logging.info("✅ Embeddings ready.")
                st.write("✅ Embeddings Done.")

                # Step 4: Build ChromaDB index
                #vector_store_mgr = ChromaVectorStore(chunks, embeddings)
                vector_store_mgr = ChromaVectorStore(documents, embeddings)
                vector_db = vector_store_mgr.create_vectorstore()
                logging.info("✅ ChromaDB index built.")
                st.write("✅ ChromaDB vector store created successfully")

                # Step 5: Configure retriever
                retriever_mgr = Retriever(vector_db, top_k = user_controls["TOP_K"])
                vector_retriever = retriever_mgr.get_retriever()
                st.write("✅ vector_retriever created successfully")

                # Store in session state for reuse across interactions
                st.session_state["vector_retriever"] = vector_retriever

                # Step 6: Build and cache QAChain
                qa_chain = QAChain(
                        retriever = vector_retriever,
                        groq_api_key = user_controls["GROQ_API_KEY"],
                        model_name = user_controls["LLM_MODEL"],
                        temperature = user_controls["TEMPERATURE"],
                        max_tokens = user_controls["TOKEN"],
                    )
                st.session_state["qa_chain"] = qa_chain

                logging.info("QAChain built and cached in session state.")

                logging.info("RAG ingestion pipeline completed successfully.")

        except Exception as e:
            logging.exception("Ingestion pipeline failed.")
            raise CustomException(e, sys)
    
    # -----------------------------------------------------------------------
    # AUTOSAR Context Retrieval
    # -----------------------------------------------------------------------
    def retrieve_context(self, user_query: str, user_controls: dict) -> str:
        """
        Retrieve the most relevant document chunks for a user query.

        Args:
            user_query (str): The user's input query.
            user_controls (dict): UI configuration dict.

        Returns:
            str: Concatenated top-K document chunks, or empty string
                 if no documents have been indexed.

        Raises:
            CustomException: If retrieval fails.
        """

        retriever = st.session_state.get("vector_retriever")

        if retriever is None:
            logging.warning("No documents indexed. Context will be empty.")
            return ""

        try:
            logging.info("Retrieving context for user query.")
            qa_chain: Optional[QAChain] = st.session_state.get("qa_chain")

            if qa_chain is None:
                logging.info("QAChain not cached. Building now...")
                qa_chain = QAChain(
                        retriever = retriever,
                        groq_api_key = user_controls["GROQ_API_KEY"],
                        model_name = user_controls["LLM_MODEL"],
                        temperature = user_controls["TEMPERATURE"],
                        max_tokens = user_controls["TOKEN"]
                    )
                
                st.session_state["qa_chain"] = qa_chain

            context = qa_chain.run(user_query)
            logging.info("Context retrieved: %d characters.", len(context))
            return context
        
        except Exception as e:
            logging.exception("Context retrieval failed.")
            raise CustomException(e, sys)