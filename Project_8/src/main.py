import sys
import warnings
import streamlit as st
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.ui.streamlit_app import StreamlitApp
warnings.filterwarnings("ignore")

class MultiModalRAGSystem:
    """
    Generic Multi-Modal RAG System.

    This application allows users to upload ANY PDF document and
    ask natural language questions about the uploaded content.

    Supported PDF Content:
        - Research Papers
        - Financial Reports
        - Technical Documents
        - Legal Documents
        - Medical Reports
        - Books
        - Notes
        - Invoices
        - Contracts
        - Study Materials

    Supported Modalities:
        - Text
        - Tables
        - Images
        - Charts
        - Figures

    Features:
        - Multi-modal retrieval
        - Hybrid semantic search
        - Context-grounded QA
        - Pinecone vector search
        - CLIP embeddings
        - Groq LLM integration
    """

    def __init__(self) -> None:
        """
        Initialize the Multi-Modal RAG System.
        """

        try:
            logging.info("Initializing Multi-Modal RAG System.")

            # Initialize Streamlit App
            self._app = StreamlitApp()

            # Load UI Controls
            self._user_input = self._app.load_streamlit_ui()
            
            logging.info("Multi-Modal RAG System initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize Multi-Modal RAG System.")
            raise CustomException(e, sys)

    # MAIN APPLICATION WORKFLOW
    def run(self) -> None:
        """
        Execute the complete application workflow.

        Workflow:
            1. User uploads PDF
            2. PDF ingestion pipeline runs
            3. Multi-modal embeddings created
            4. User asks questions
            5. Relevant context retrieved
            6. LLM generates grounded response
        """

        try:
            # PAGE TITLE
            st.title(
                "📚 Multi-Modal RAG System"
            )

            st.markdown(
                """
                Upload any PDF document and ask questions about:
                - Text
                - Tables
                - Charts
                - Figures
                - Images
                """
            )

            # --------------------------------------------------
            # PDF UPLOAD
            # --------------------------------------------------
            uploaded_file = st.sidebar.file_uploader(
                label="📂 Upload PDF Document",
                type=["pdf"],
                accept_multiple_files=False,
                help=(
                    "Upload any PDF document for "
                    "multi-modal question answering."
                ),
            )

            # --------------------------------------------------
            # INGESTION PIPELINE
            # --------------------------------------------------
            if uploaded_file:

                # Prevent reprocessing same file
                current_file_name = uploaded_file.name

                previously_uploaded = (
                    st.session_state.get(
                        "uploaded_file_name"
                    )
                )

                if (
                    previously_uploaded
                    != current_file_name
                ):

                    with st.spinner(
                        "📄 Processing PDF document..."
                    ):

                        self._app.run_ingestion_pipeline(
                            uploaded_file,
                            self._user_input,
                        )

                    # Save uploaded filename
                    st.session_state[
                        "uploaded_file_name"
                    ] = current_file_name

                    st.sidebar.success(
                        "✅ PDF processed successfully."
                    )

                    logging.info(
                        "PDF ingestion completed successfully."
                    )

                else:

                    st.sidebar.info(
                        "✅ PDF already processed."
                    )

            else:

                st.sidebar.info(
                    "Upload a PDF document to begin."
                )

                logging.info(
                    "Waiting for PDF upload."
                )

            # --------------------------------------------------
            # CHAT INTERFACE
            # --------------------------------------------------
            user_query = st.chat_input(
                "Ask questions about the uploaded PDF..."
            )

            if user_query:

                # Display user message
                with st.chat_message("user"):
                    st.write(user_query)

                self._run_query_pipeline(
                    user_query=user_query.strip()
                )

        except Exception as e:
            logging.exception(
                "Application workflow execution failed."
            )
            raise CustomException(e, sys)

    # ==========================================================
    # QUERY + RETRIEVAL PIPELINE
    # ==========================================================
    def _run_query_pipeline(
        self,
        user_query: str,
    ) -> None:
        """
        Execute retrieval + response generation pipeline.

        Args:
            user_query (str):
                User question.
        """

        try:
            logging.info(
                f"Executing query: {user_query}"
            )

            # --------------------------------------------------
            # CHECK DOCUMENT AVAILABILITY
            # --------------------------------------------------
            retriever = st.session_state.get(
                "vector_retriever"
            )

            if retriever is None:

                st.warning(
                    "⚠️ Please upload and process a PDF document first."
                )

                logging.warning(
                    "No retriever found in session state."
                )

                return

            # --------------------------------------------------
            # RETRIEVE CONTEXT + GENERATE RESPONSE
            # --------------------------------------------------
            with st.spinner(
                "🔍 Retrieving relevant content..."
            ):

                response = (
                    self._app.retrieve_multimodal_context(
                        user_query,
                        self._user_input,
                    )
                )

            # --------------------------------------------------
            # DISPLAY RESPONSE
            # --------------------------------------------------
            with st.chat_message("assistant"):

                st.write(response)

            logging.info(
                "Response generated successfully."
            )

        except Exception as e:
            logging.exception(
                "Query pipeline failed."
            )
            raise CustomException(e, sys)