"""
Streamlit Frontend for Agentic RAG Knowledge Assistant.

This module provides the Streamlit-based frontend that communicates
with the FastAPI backend for document upload and query processing.

Features:
    - Upload PDF documents to the backend.
    - Ask questions and receive AI-generated responses.
    - Display chat history.
    - Configure LLM settings via sidebar.
"""

import sys
import requests
import streamlit as st
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.config.settings import Config
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────

FASTAPI_BASE_URL = "http://localhost:8000"


class StreamlitFrontend:
    """
    Streamlit frontend that communicates with the FastAPI backend.

    Responsibilities:
        - Render Streamlit UI components.
        - Manage user configuration settings.
        - Upload PDFs to the FastAPI backend.
        - Submit queries to the FastAPI backend.
        - Display AI-generated responses.
    """

    def __init__(self):
        """Initialize Streamlit frontend."""
        try:
            logging.info("Initializing Streamlit Frontend.")
            self._config = Config()
            logging.info("Streamlit Frontend initialized successfully.")
        except Exception as e:
            logging.exception("Failed to initialize Streamlit Frontend.")
            raise CustomException(e, sys)

    def run(self):
        """Main entry point to run the Streamlit frontend application."""
        try:
            self._setup_page()
            user_controls = self._render_sidebar()
            self._render_chat_history()
            self._handle_document_upload(user_controls)
            self._handle_user_query(user_controls)

        except Exception as e:
            logging.exception("Error running Streamlit Frontend.")
            raise CustomException(e, sys)

    def _setup_page(self):
        """Configure Streamlit page settings."""
        page_title = "📚 " + self._config.get_page_title()
        st.set_page_config(page_title=page_title, layout="wide")
        st.header(page_title)

    def _render_sidebar(self) -> dict:
        """
        Render sidebar configuration panel.

        Returns:
            dict: User configuration settings.
        """
        user_controls = {}

        # Load API key from environment/secrets
        groq_api_key = st.secrets.get("GROQ_API_KEY", "")

        if not groq_api_key:
            st.warning(
                "⚠️ GROQ_API_KEY not found. Enter your Groq API key in Streamlit secrets."
            )

        with st.sidebar:
            st.subheader("⚙️ Configuration")

            # API Key status
            st.text_input(
                "🔑 Groq API Key:",
                value="✅ Configured" if groq_api_key else "❌ Not Set",
                disabled=True,
                key="GROQ_API_KEY_DISPLAY",
            )

            # LLM Model selection
            selected_llm = st.selectbox(
                "🧠 Select LLM Model",
                self._config.get_groq_model_options(),
            )
            user_controls["LLM_MODEL"] = selected_llm

            # Temperature slider
            temp_options = self._config.get_temperature()
            selected_temperature = st.slider(
                "🔥 Temperature:",
                min_value=temp_options[0],
                max_value=temp_options[-1],
                value=temp_options[1],
            )
            user_controls["TEMPERATURE"] = selected_temperature

            # Max Tokens slider
            token_options = self._config.get_token()
            selected_token = st.slider(
                "📏 Max Tokens:",
                min_value=token_options[0],
                max_value=token_options[-1],
                value=token_options[1],
            )
            user_controls["TOKEN"] = selected_token

            # Backend status
            st.divider()
            self._display_backend_status()

        user_controls["GROQ_API_KEY"] = groq_api_key
        return user_controls

    def _display_backend_status(self):
        """Display the connection status with the FastAPI backend."""
        try:
            response = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.success("🟢 Backend Connected")
                if data.get("documents_indexed"):
                    st.info("📄 Documents indexed and ready")
                else:
                    st.info("📂 No documents indexed yet")
            else:
                st.warning("⚠️ Backend returned unexpected status.")
        except requests.ConnectionError:
            st.warning("⏳ Backend is starting up... Please refresh in a few seconds.")
        except Exception:
            st.warning("⏳ Backend is initializing...")

    def _render_chat_history(self):
        """Display existing chat history."""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _handle_document_upload(self, user_controls: dict):
        """
        Handle PDF document upload via sidebar.

        Sends files to FastAPI /upload endpoint.
        """
        with st.sidebar:
            st.divider()
            st.subheader("📂 Document Upload")

            uploaded_files = st.file_uploader(
                "Upload PDF(s)",
                type=["pdf"],
                accept_multiple_files=True,
                help="Select one or more PDFs to index.",
            )

            index_clicked = st.button(
                "⚡ Index PDF(s)",
                disabled=(not uploaded_files),
                use_container_width=True,
                type="primary",
            )

        if uploaded_files and index_clicked:
            self._upload_to_backend(uploaded_files, user_controls)

    def _upload_to_backend(self, uploaded_files, user_controls: dict):
        """
        Upload files to the FastAPI backend.

        Args:
            uploaded_files: List of Streamlit UploadedFile objects.
            user_controls: User configuration settings.
        """
        with st.sidebar:
            with st.spinner("🔎 Processing PDFs..."):
                try:
                    # Prepare multipart form data
                    files_data = []
                    for file in uploaded_files:
                        file.seek(0)
                        files_data.append(
                            ("files", (file.name, file.read(), "application/pdf"))
                        )

                    # Prepare form fields
                    form_data = {
                        "llm_model": user_controls.get("LLM_MODEL", ""),
                        "temperature": str(user_controls.get("TEMPERATURE", 0.2)),
                        "max_tokens": str(user_controls.get("TOKEN", 800)),
                    }

                    if user_controls.get("GROQ_API_KEY"):
                        form_data["groq_api_key"] = user_controls["GROQ_API_KEY"]

                    # Send to FastAPI backend
                    response = requests.post(
                        f"{FASTAPI_BASE_URL}/upload",
                        files=files_data,
                        data=form_data,
                        timeout=300,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(
                            f"✅ {result['message']} "
                            f"({result['pages_extracted']} pages from "
                            f"{len(result['files_processed'])} file(s))"
                        )
                        logging.info("Documents uploaded and indexed successfully.")
                    else:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"❌ Upload failed: {error_detail}")
                        logging.error("Upload failed: %s", error_detail)

                except requests.ConnectionError:
                    st.error(
                        "❌ Cannot connect to backend. Ensure FastAPI server is running."
                    )
                    logging.error("Backend connection failed during upload.")
                except Exception as e:
                    st.error(f"❌ Upload error: {str(e)}")
                    logging.exception("Upload to backend failed.")

    def _handle_user_query(self, user_controls: dict):
        """
        Handle user query input and display response.

        Args:
            user_controls: User configuration settings.
        """
        user_query = st.chat_input("💬 Enter your question here...")

        if user_query:
            # Display user message
            st.session_state["messages"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Get response from backend
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching & generating answer..."):
                    response_text = self._query_backend(user_query.strip(), user_controls)
                    st.markdown(response_text)

            # Store response in history
            st.session_state["messages"].append(
                {"role": "assistant", "content": response_text}
            )

    def _query_backend(self, question: str, user_controls: dict) -> str:
        """
        Send query to FastAPI backend and return the response.

        Args:
            question: User's question.
            user_controls: User configuration settings.

        Returns:
            str: Formatted response string.
        """
        try:
            payload = {
                "question": question,
                "llm_model": user_controls.get("LLM_MODEL"),
                "temperature": user_controls.get("TEMPERATURE", 0.2),
                "max_tokens": user_controls.get("TOKEN", 800),
            }

            response = requests.post(
                f"{FASTAPI_BASE_URL}/query",
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "No response generated.")
                tool_used = result.get("tool_used")

                if tool_used:
                    return f"{answer}\n\n[🛠️ Tool Used: {tool_used}]"
                return answer

            else:
                error_detail = response.json().get("detail", "Unknown error")
                logging.error("Query failed: %s", error_detail)
                return f"❌ Error: {error_detail}"

        except requests.ConnectionError:
            logging.error("Backend connection failed during query.")
            return "❌ Cannot connect to backend. Ensure FastAPI server is running."
        except requests.Timeout:
            logging.error("Backend request timed out.")
            return "❌ Request timed out. The backend might be processing a heavy request."
        except Exception as e:
            logging.exception("Query to backend failed.")
            return f"❌ Error: {str(e)}"
