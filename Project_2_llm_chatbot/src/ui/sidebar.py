import streamlit as st
from src.config.settings import AppSettings

class Sidebar:
    """Sidebar UI component."""

    def render(self) -> dict:
        st.sidebar.header("⚙️ Configuration")

        # Load API keys from Streamlit secrets
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        model = st.sidebar.selectbox("🧠 LLM Model", AppSettings.AVAILABLE_MODELS)
        temperature = st.sidebar.slider("🔥 Temperature", 0.0, 1.0, AppSettings.DEFAULT_TEMPERATURE)
        max_tokens = st.sidebar.slider("📏 Max Tokens", AppSettings.MIN_TOKENS, AppSettings.MAX_TOKENS, AppSettings.DEFAULT_MAX_TOKENS)

        return {
            "api_key": groq_api_key,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
