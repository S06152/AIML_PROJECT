import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from src.config.settings import Config
from src.ingestion.pdf_loader import PDFLoader
from src.embedding.embedding import EmbeddingManager
from src.vectorstore.chroma_store import ChromaVectorStore
from src.tools.retriever_tool import RetrieverTool
import warnings
warnings.filterwarnings("ignore")

class StreamlitApp:
    """
    Streamlit UI controller for the Agentic RAG application.

    Responsibilities:
        - Render Streamlit UI components.
        - Manage user configuration settings.
        - Handle PDF document ingestion.
        - Create embeddings and vector store.
        - Configure retriever and store it in session state.
    """

    def __init__(self):
        """
        Initialize Streamlit application.

        Raises:
            CustomException: If initialization fails.
        """
        try:
            logging.info("Initializing Streamlit application.")
            self._config = Config()
            self._user_control: dict = {}

            logging.info("StreamlitApp initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize Streamlit application.")
            raise CustomException(e, sys)

    def load_streamlit_ui(self):
        """
        Render Streamlit page and sidebar configuration.

        Returns:
            dict: User configuration settings.
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

            logging.info("Streamlit UI loaded  successfully.")
            return self._user_control

        except Exception as e:
            logging.exception("Failed to load Streamlit UI.")
            raise CustomException(e, sys)
    
    # RAG Ingestion Pipeline
    def run_ingestion_pipeline(self, uploaded_file, user_controls: dict) -> None:
        """
        Execute the document ingestion pipeline.

        Steps:
            1. Load PDF documents.
            2. Generate embeddings.
            3. Create Chroma vector store.
            4. Configure retriever.
            5. Store retriever in session state.

        Args:
            uploaded_files: Uploaded PDF files.
            user_controls: User-selected configuration.
        """
        try:
            with st.sidebar.spinner("🔎 Processing PDFs document...."):
                logging.info("Starting document ingestion pipeline.")

                all_documents = []

                # Step 1: Load PDFs
                for file in uploaded_file:
                    try:
                        file.seek(0)
                        loader = PDFLoader(file, user_controls)
                        documents = loader.load_documents()
                        all_documents.extend(documents)
                        logging.info("PDF loaded successfully | File=%s | Pages=%s", file.name, len(documents))

                    except Exception as e:
                        logging.error(f"Failed to load {file.name}: {e}")
                        st.error(f"❌ Failed to load {file.name}")
                        continue
                st.sidebar.write("✅ PDF loaded successfully. Extracted {} pages.".format(len(all_documents)))
                
                # Step 2: Create Embeddings
                embedding_mgr = EmbeddingManager(user_controls["EMBEDDING_MODELS"])
                embeddings = embedding_mgr.create_embeddings()
                logging.info("Embeddings created successfully.")
                st.sidebar.write("✅ Embeddings generated successfully.")

                # Step 3: Create Vector Store
                vector_store_mgr = ChromaVectorStore(all_documents, embeddings)
                vector_db = vector_store_mgr.create_vectorstore()
                logging.info("Chroma vector store created successfully.")
                st.sidebar.write( "✅ ChromaDB vector store created successfully.")

                # Step 4: Create Retriever
                retriever_mgr = RetrieverTool(vector_db, top_k = user_controls["TOP_K"])
                vector_retriever = retriever_mgr.get_retriever()
                st.sidebar.write("✅ Retriever created successfully.")

                # Step 5: Store Retriever
                st.session_state["vector_retriever"] = vector_retriever
                logging.info("Retriever stored in session state.")
                st.sidebar.success("✅ Documents indexed successfully.")

                logging.info("Document ingestion pipeline completed successfully.")

        except Exception as e:
            logging.exception("Document ingestion pipeline failed.")
            raise CustomException(e, sys)