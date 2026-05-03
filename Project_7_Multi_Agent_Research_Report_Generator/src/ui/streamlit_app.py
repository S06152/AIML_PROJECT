# Standard Library Imports
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from src.config.settings import Config
import warnings
warnings.filterwarnings("ignore")

class StreamlitApp:
    """
    Main Streamlit UI controller for the Multi-Agent Research & Report Generator.

    Responsibilities:
        - Load and display sidebar configuration controls
        - Capture user configuration for LLM and pipeline
        - Provide inputs for multi-agent workflow execution
    """

    def __init__(self):
        """
        Initialize StreamlitApp: load configuration and prepare state.

        Raises:
            CustomException: If initialization fails
        """
        try:
            # Load configuration settings
            logging.info("Loading application configuration...")
            self._config = Config()

            # Dictionary to store user-selected controls
            self._user_control: dict = {}

            logging.info("StreamlitApp initialized successfully.")

        except Exception as e:
            logging.exception("ERROR during StreamlitApp initialization.")
            raise CustomException(e, sys)

    # MAIN UI LOADER
    def load_streamlit_ui(self):
        """
        Render the Streamlit UI and sidebar configuration.

        Returns:
            dict: User-selected configuration values:
                - GROQ_API_KEY
                - LLM_MODEL
                - TEMPERATURE
                - TOKEN
        Raises:
             CustomException: If UI rendering fails
        """
        try:
            logging.info("Loading Streamlit UI.")

            # Page Configuration
            page_title = "🤖  " + self._config.get_page_title()
            st.set_page_config(page_title = page_title, layout = "wide")
            st.header(page_title)

            # Load API keys
            logging.info("Fetching GROQ API key from Streamlit secrets...")
            groq_api_key = st.secrets.get("GROQ_API_KEY")
            
            if not groq_api_key:
                logging.warning("GROQ_API_KEY not found in Streamlit secrets.")
                st.warning("⚠️ GROQ_API_KEY not found. Please add it in Streamlit secrets to enable LLM functionality.")
            else:
                logging.info("GROQ_API_KEY loaded successfully.")

            # Sidebar Configuration
            logging.info("Rendering sidebar configuration...")
            with st.sidebar:
                st.subheader("⚙️ Configuration")

                # API Key status (masked / read-only)
                st.text_input("🔑 Groq API Key:", value = "✅ Configured" if groq_api_key else "❌ Not Set", disabled = True, key = "GROQ_API_KEY")

                # LLM Model Selection
                selected_llm = st.selectbox("🧠 Select LLM Model", self._config.get_groq_model_options())
                self._user_control["LLM_MODEL"] = selected_llm
                logging.info(f"Selected LLM Model: {selected_llm}")

                # Temperature Selection
                temp_options = self._config.get_temperature()
                selected_temperature = st.slider("🔥 Temperature:", min_value = temp_options[0], max_value = temp_options[-1], value = temp_options[1])
                self._user_control["TEMPERATURE"] = selected_temperature
                logging.info(f"Selected Temperature: {selected_temperature}")

                # Token Limit Selection
                token_options = self._config.get_token()
                selected_token = st.slider("📏 Max Tokens:", min_value = token_options[0], max_value = token_options[-1], value = token_options[1])
                self._user_control["TOKEN"] = selected_token
                logging.info(f"Selected Max Tokens: {selected_token}")
            
           # Store API Key
            self._user_control["GROQ_API_KEY"] = groq_api_key

            logging.info("Streamlit UI rendered successfully.")
            return self._user_control

        except Exception as e:
            logging.exception("CRITICAL ERROR while rendering Streamlit UI.")
            raise CustomException(e, sys)