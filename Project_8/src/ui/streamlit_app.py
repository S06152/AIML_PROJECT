# Standard Library Imports
import sys
import warnings
from typing import Optional
import os

# Third-Party Imports
import numpy as np
import streamlit as st

# Custom Imports
from src.utils.logger import logging
from src.utils.exception import CustomException

from src.config.settings import Config

from src.ingestion.pdf_loader import PDFLoader
from src.chunking.chunk import ChunkingStrategy
from src.embedding.embedding import EmbeddingManager
from src.vectorstore.pinecone_store import PineconeVector
from src.retrieval.retriever import Retriever
from src.chain.qa_chain import QAChain

warnings.filterwarnings("ignore")


class StreamlitApp:
    """
    Main Streamlit Controller for the Multi-Modal RAG System.

    Responsibilities:
        - Render Streamlit UI
        - Upload and process PDFs
        - Execute ingestion pipeline
        - Create embeddings
        - Store vectors in Pinecone
        - Perform hybrid retrieval
        - Run QA pipeline
        - Display grounded responses

    Supported Modalities:
        - Text
        - Tables
        - Images
        - Charts
        - Figures

    Example Queries:
        - "What does Figure 3 show?"
        - "Summarize the paper."
        - "Explain the table on page 5."
    """

    def __init__(self) -> None:
        """
        Initialize Streamlit Application.

        Raises:
            Exception:
                If initialization fails.
        """

        try:
            logging.info(
                "Initializing StreamlitApp."
            )

            self._config = Config()

            self._user_control = {}

            logging.info(
                "StreamlitApp initialized successfully."
            )

        except Exception as e:
            logging.exception(
                "Failed to initialize StreamlitApp."
            )
            raise CustomException(e, sys)

    # ==========================================================
    # STREAMLIT UI
    # ==========================================================
    def load_streamlit_ui(self) -> dict:
        """
        Render Streamlit UI and sidebar configuration.

        Returns:
            dict:
                User-selected configuration values.
        """

        try:
            logging.info(
                "Loading Streamlit UI."
            )

            page_title = (
                "📄 Multi-Modal RAG System"
            )

            st.set_page_config(
                page_title=page_title,
                layout="wide",
            )

            st.title(page_title)

            st.markdown(
                """
                Upload research papers or reports containing:

                - Text
                - Tables
                - Charts
                - Figures
                - Images

                Ask questions like:
                **"What does Figure 3 show?"**
                """
            )

            # --------------------------------------------------
            # Load API Keys
            # --------------------------------------------------
            groq_api_key = st.secrets.get(
                "GROQ_API_KEY"
            )

            pinecone_api_key = st.secrets.get(
                "PINECONE_API_KEY"
            )

            # --------------------------------------------------
            # Sidebar Configuration
            # --------------------------------------------------
            with st.sidebar:

                st.header(
                    "⚙️ Configuration"
                )

                st.text_input(
                    "Groq API Key",
                    value=(
                        "✅ Configured"
                        if groq_api_key
                        else "❌ Not Configured"
                    ),
                    disabled=True,
                )

                st.text_input(
                    "Pinecone API Key",
                    value=(
                        "✅ Configured"
                        if pinecone_api_key
                        else "❌ Not Configured"
                    ),
                    disabled=True,
                )

                # ------------------------------
                # LLM Configuration
                # ------------------------------
                selected_model = st.selectbox(
                    "🧠 Select LLM",
                    self._config.get_groq_model_options(),
                )

                selected_temperature = st.slider(
                    "🔥 Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.1,
                )

                selected_tokens = st.slider(
                    "📏 Max Tokens",
                    min_value=256,
                    max_value=4096,
                    value=1024,
                    step=128,
                )

                # ------------------------------
                # Retrieval Configuration
                # ------------------------------
                top_k = st.slider(
                    "🔍 Top-K Retrieval",
                    min_value=1,
                    max_value=10,
                    value=5,
                )

                chunk_size = st.slider(
                    "✂️ Chunk Size",
                    min_value=256,
                    max_value=2000,
                    value=1000,
                    step=100,
                )

                chunk_overlap = st.slider(
                    "🔁 Chunk Overlap",
                    min_value=0,
                    max_value=500,
                    value=200,
                    step=50,
                )

            # --------------------------------------------------
            # Store User Configurations
            # --------------------------------------------------
            self._user_control = {
                "GROQ_API_KEY": groq_api_key,
                "PINECONE_API_KEY": pinecone_api_key,
                "LLM_MODEL": selected_model,
                "TEMPERATURE": selected_temperature,
                "TOKEN": selected_tokens,
                "TOP_K": top_k,
                "CHUNK_SIZE": chunk_size,
                "CHUNK_OVERLAP": chunk_overlap,
                "EMBEDDING_MODEL": (
                    "openai/clip-vit-base-patch32"
                ),
            }

            logging.info(
                "Streamlit UI rendered successfully."
            )

            return self._user_control

        except Exception as e:
            logging.exception(
                "Error while rendering Streamlit UI."
            )
            raise CustomException(e, sys)

    # ==========================================================
    # INGESTION PIPELINE
    # ==========================================================
    def run_ingestion_pipeline(
        self,
        uploaded_file,
        user_controls: dict,
    ) -> Optional[str]:
        """
        Run complete Multi-Modal ingestion pipeline.

        Pipeline:
            1. Load PDF
            2. Extract text/tables/images
            3. Chunk documents
            4. Generate embeddings
            5. Store embeddings in Pinecone
            6. Configure retriever

        Args:
            uploaded_file:
                Streamlit UploadedFile object.

            user_controls (dict):
                User-selected configurations.

        Returns:
            Optional[str]:
                Success message.
        """

        try:
            logging.info(
                "Starting Multi-Modal ingestion pipeline."
            )

            # --------------------------------------------------
            # STEP 1: LOAD PDF
            # --------------------------------------------------
            loader = PDFLoader(uploaded_file)

            documents = loader.load_documents()

            logging.info(
                f"Loaded {len(documents)} "
                f"multi-modal documents."
            )

            # --------------------------------------------------
            # STEP 2: CHUNK DOCUMENTS
            # --------------------------------------------------
            chunker = ChunkingStrategy(
                chunk_size=user_controls[
                    "CHUNK_SIZE"
                ],
                chunk_overlap=user_controls[
                    "CHUNK_OVERLAP"
                ],
            )

            chunks, embeddings = chunker.split_documents_into_chunks(documents)
            logging.info(f"Generated {len(chunks)} chunks.")
            for i, chunk in enumerate(chunks):
                logging.info(f"Chunk {i}: {chunk.page_content[:100]} ... Metadata: {chunk.metadata}")

            # --------------------------------------------------
            # STEP 3: PREPARE PINECONE DATA
            # --------------------------------------------------
            vector_records = []

            for doc, embedding in zip(chunks, embeddings):
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                vector_records.append(
                    {
                        "text": doc.page_content,
                        "embedding": embedding,
                        "metadata": doc.metadata,
                    }
                )
            logging.info(f"Prepared {len(vector_records)} vector records for Pinecone.")
            for i, record in enumerate(vector_records):
                logging.info(f"Vector {i}: Embedding length: {len(record['embedding'])}, Text: {record['text'][:100]} ...")

            # --------------------------------------------------
            # STEP 4: STORE IN PINECONE
            # --------------------------------------------------
            vector_store_manager = PineconeVector(
                vector_records=vector_records,
                api_key=user_controls[
                    "PINECONE_API_KEY"
                ],
            )

            vector_db = (
                vector_store_manager.create_vectorstore()
            )

            logging.info(
                "Pinecone vector store created successfully."
            )

            # --------------------------------------------------
            # STEP 5: CONFIGURE RETRIEVER
            # --------------------------------------------------
            retriever_manager = Retriever(
                vector_store=vector_db,
                top_k=user_controls["TOP_K"],
            )

            vector_retriever = (
                retriever_manager.get_retriever()
            )

            # --------------------------------------------------
            # STORE SESSION STATE
            # --------------------------------------------------
            st.session_state[
                "vector_retriever"
            ] = vector_retriever

            st.session_state[
                "retriever_manager"
            ] = retriever_manager

            logging.info(
                "Retriever stored in session state."
            )

            # --------------------------------------------------
            # STEP 6: INITIALIZE QA CHAIN
            # --------------------------------------------------
            qa_chain = QAChain(
                retriever=vector_retriever,
                groq_api_key=user_controls[
                    "GROQ_API_KEY"
                ],
                model_name=user_controls[
                    "LLM_MODEL"
                ],
                temperature=user_controls[
                    "TEMPERATURE"
                ],
                max_tokens=user_controls[
                    "TOKEN"
                ],
            )

            st.session_state["qa_chain"] = (
                qa_chain
            )

            logging.info(
                "QAChain initialized successfully."
            )

            success_message = (
                "✅ Multi-Modal document ingestion "
                "completed successfully."
            )

            logging.info(success_message)

            return success_message

        except Exception as e:
            logging.exception(
                "Ingestion pipeline failed."
            )
            raise CustomException(e, sys)

    # ==========================================================
    # RETRIEVE MULTI-MODAL CONTEXT
    # ==========================================================
    def retrieve_multimodal_context(
        self,
        user_query: str,
        user_controls: dict,
    ) -> str:
        """
        Retrieve relevant multi-modal context.

        Args:
            user_query (str):
                User query.

            user_controls (dict):
                Configuration dictionary.

        Returns:
            str:
                Retrieved grounded response.
        """

        try:
            retriever = st.session_state.get(
                "vector_retriever"
            )

            if retriever is None:

                logging.warning(
                    "No retriever found in session state."
                )

                return ""

            logging.info(
                f"Retrieving multi-modal context "
                f"for query: {user_query}"
            )

            qa_chain = st.session_state.get(
                "qa_chain"
            )

            if qa_chain is None:

                qa_chain = QAChain(
                    retriever=retriever,
                    groq_api_key=user_controls[
                        "GROQ_API_KEY"
                    ],
                    model_name=user_controls[
                        "LLM_MODEL"
                    ],
                    temperature=user_controls[
                        "TEMPERATURE"
                    ],
                    max_tokens=user_controls[
                        "TOKEN"
                    ],
                )

                st.session_state["qa_chain"] = (
                    qa_chain
                )

            response = qa_chain.run(user_query)

            logging.info(
                "Multi-modal response generated successfully."
            )

            return response

        except Exception as e:
            logging.exception(
                "Failed during multi-modal retrieval."
            )
            raise CustomException(e, sys)

    # ==========================================================
    # GENERATE RESPONSE
    # ==========================================================
    def generate_response(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Return final response.

        Args:
            query (str):
                User query.

            context (str):
                Retrieved/generated response.

        Returns:
            str:
                Final formatted response.
        """

        try:
            logging.info(
                "Generating final response."
            )

            return context

        except Exception as e:
            logging.exception(
                "Error while generating response."
            )
            raise CustomException(e, sys)

# Add this at the bottom of your main Streamlit script (e.g., app.py or streamlit_app.py)
log_file_path = "app.log"  # Update this if your log file has a different name or path
if os.path.exists(log_file_path):
    with open(log_file_path, "rb") as log_file:
        st.download_button(
            label="Download Log File",
            data=log_file,
            file_name="app.log",
            mime="text/plain"
        )
else:
    st.info("Log file not found.")