import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from src.config.settings import Config
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
            tavily_api_key = st.secrets.get("TAVILY_API_KEY")
            
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
            self._user_control["TAVILY_API_KEY"] = tavily_api_key
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