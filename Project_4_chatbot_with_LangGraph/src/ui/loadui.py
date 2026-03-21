import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from .uiconfigfile import Config

class LoadStreamlitUI:
    """
    This class handles:
    - Loading UI configuration from Config class
    - Rendering Streamlit components
    - Capturing user selections
    """

    def __init__(self):
        """
        Constructor:
        - Initializes configuration reader
        - Creates dictionary to store user inputs
        """
        try:
            logging.info("Initializing LoadStreamlitUI")
            self.config = Config()
            self.user_controls = {}

        except Exception as e:
            logging.error(f"Error during UI initialization: {str(e)}")
            raise CustomException(e, sys)

    def load_streamlit_ui(self):
        """
        Builds and renders the Streamlit UI.

        Returns:
            dict: Dictionary containing user-selected values
        """
        try:
            logging.info("Loading Streamlit UI")

            # Set page configuration (title + layout)
            page_title = "🤖 " + self.config.get_page_title()
            st.set_page_config(page_title = page_title, layout = "wide")

            # Display main header
            st.header(page_title)

            # Sidebar section
            with st.sidebar:
                logging.info("Rendering sidebar components")

                # API Key input (from Streamlit secrets)
                groq_api_key = st.secrets.get("GROQ_API_KEY")
                self.user_controls["GROQ_API_KEY"] = groq_api_key

                # Model selection dropdown
                model_options = self.config.get_groq_model_options()

                if not model_options:
                    logging.warning("No model options found in config")
                    st.error("No models available. Please check configuration.")
                    return self.user_controls

                selected_model = st.selectbox("Select Model", model_options)
                self.user_controls["selected_groq_model"] = selected_model

                logging.info(f"Selected model: {selected_model}")

                # Validate API key
                if not groq_api_key:
                    logging.warning("GROQ API key not found in Streamlit secrets")
                    st.warning(
                        "⚠️ Please add your GROQ API key in Streamlit secrets.\n"
                        "Get it from: https://console.groq.com/keys"
                    )

            logging.info("Streamlit UI loaded successfully")
            return self.user_controls

        except Exception as e:
            logging.error(f"Error while loading Streamlit UI: {str(e)}")
            raise CustomException(e, sys)