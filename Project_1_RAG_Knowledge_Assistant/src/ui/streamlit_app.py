# Standard Library Imports
import sys
import os
#from dotenv import load_dotenv
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from src.config.settings import Config
from src.ingestion.loader_factory import LoaderFactory
from src.chunking.chunk import ChunkingStrategy
from src.embedding.embedding import EmbeddingManager
from src.vectorstore.store_factory import VectorStoreFactory
from src.retrieval.retriever import Retriever
from src.chain.qa_chain import QAChain

class StreamlitApp:
    """
    Main Streamlit UI controller for the RAG Knowledge Assistant.

    Responsibilities:
    - Manage UI rendering
    - Handle file ingestion & indexing
    - Manage session state
    - Run Retrieval + QA chain
    """

    def __init__(self):
        try:
            logging.info("Initializing StreamlitApp.")
            #load_dotenv()
            self.config = Config()

        except Exception as e:
            logging.exception("Error initializing StreamlitApp.")
            raise CustomException(e, sys)

    def _get_files_signature(self, uploaded_files):
        """
        Generate unique signature for uploaded files.
        Used to detect file changes for auto re-indexing.
        """
        if not uploaded_files:
            return None
        
        return tuple(sorted((f.name, f.size) for f in uploaded_files))

    def _needs_reindexing(self, uploaded_files, selected_vector_db):
        """
        Check whether documents or vector DB selection changed.
        """
        current_sig = self._get_files_signature(uploaded_files)

        if current_sig is None:
            return False

        prev_sig = st.session_state.get("_files_signature")
        prev_db = st.session_state.get("_selected_vector_db")

        needs_update = (
            "vector_store" not in st.session_state
            or current_sig != prev_sig
            or selected_vector_db != prev_db
        )

        if needs_update:
            logging.info("Reindexing triggered due to file or DB change.")

        return needs_update

    def _validate_vector_db_keys(self, selected_vector_db, pinecone_api_key, astra_db_token, astra_db_endpoint):
        """
        Validate that required API keys are provided for the selected vector DB.
        Returns (is_valid, error_message).
        """
        if "pinecone" in selected_vector_db.lower():
            if not pinecone_api_key:
                return False, "❌ Pinecone API Key is required for Pinecone vector store."
        elif "astradb" in selected_vector_db.lower():
            if not astra_db_token:
                return False, "❌ AstraDB Application Token is required for AstraDB vector store."
            if not astra_db_endpoint:
                return False, "❌ AstraDB API Endpoint is required for AstraDB vector store."
        return True, ""

    # DOCUMENT PROCESSING PIPELINE
    def _process_and_index(self, uploaded_files, selected_vector_db, pinecone_api_key, astra_db_token, astra_db_endpoint):
        """
        Full ingestion pipeline:
        1. Load documents
        2. Chunk
        3. Embed
        4. Create vector store
        """

        try:
            # Validate vector DB keys before starting the pipeline
            is_valid, error_msg = self._validate_vector_db_keys(
                selected_vector_db, pinecone_api_key, astra_db_token, astra_db_endpoint
            )
            if not is_valid:
                st.error(error_msg)
                return

            with st.spinner("📄 Loading & processing documents..."):
                logging.info("Starting document ingestion pipeline.")

                # Step 1: Ingestion
                all_documents = []

                for uploaded_file in uploaded_files:
                    try:
                        uploaded_file.seek(0)
                        loader = LoaderFactory.get_loader(uploaded_file)
                        docs = loader.load_documents()
                        all_documents.extend(docs)

                        logging.info(f"Loaded {len(docs)} pages from {uploaded_file.name}")

                    except Exception as e:
                        logging.error(f"Failed to load {uploaded_file.name}: {e}")
                        st.error(f"❌ Failed to load {uploaded_file.name}")
                        continue

                if not all_documents:
                    logging.warning("No documents were successfully loaded.")
                    st.error("No documents could be loaded.")
                    return

                # Step 2: Chunking
                chunker = ChunkingStrategy(chunk_size = self.config.get_chunk_size(), chunk_overlap = self.config.get_chunk_overlap())
                chunks = chunker.split_documents_into_chunks(all_documents)

                if not chunks:
                    logging.warning("No chunks were created after splitting.")
                    st.error("❌ Document splitting produced no chunks.")
                    return

                logging.info(f"Created {len(chunks)} chunks.")

                # Step 3: Embeddings
                embedding_manager = EmbeddingManager(model_name = self.config.get_embedding_model())
                embeddings = embedding_manager.create_embeddings()

                # Step 4: API Key Selection
                if "pinecone" in selected_vector_db.lower():
                    api_key = pinecone_api_key
                elif "astradb" in selected_vector_db.lower():
                    api_key = astra_db_token
                else:
                    api_key = None

                # Step 5: Vector Store Creation
                vector_store_wrapper = VectorStoreFactory.get_vector_store(
                    vector_db_name = selected_vector_db,
                    documents = chunks,
                    embeddings = embeddings,
                    api_key = api_key,
                    api_endpoint = astra_db_endpoint,
                )

                vector_store = vector_store_wrapper.create_vectorstore()

                # Save state
                st.session_state["vector_store"] = vector_store
                st.session_state["_files_signature"] = self._get_files_signature(uploaded_files)
                st.session_state["_selected_vector_db"] = selected_vector_db
                st.session_state["_chunk_count"] = len(chunks)
                st.session_state["messages"] = []

                logging.info("Indexing completed successfully.")

            st.success(f"✅ Indexed {len(chunks)} chunks into {selected_vector_db}")

        except Exception as e:
            logging.exception("Error during document processing pipeline.")
            st.error(f"❌ Indexing failed: {e}")

    # MAIN UI LOADER
    def load_streamlit_ui(self):
        """
        Render Streamlit UI and manage chat workflow.
        """
        try:
            logging.info("Loading Streamlit UI.")

            page_title = "🤖 " + self.config.get_page_title()
            st.set_page_config(page_title = page_title, layout = "wide")
            st.header(page_title)

            # Sidebar Configuration
            with st.sidebar:
                st.subheader("⚙️ Configuration")
                #groq_api_key = st.text_input("🔑 Groq API Key:", type = "password", key = "GROQ_API_KEY", value = os.getenv("GROQ_API_KEY"), disabled = True)
                #pinecone_api_key = st.text_input("🔑 Pinecone API Key:", type = "password", key = "PINECONE_API_KEY", value = os.getenv("PINECONE"), disabled = True)
                #astra_db_token = st.text_input("🔑 AstraDB Application Token:", type = "password", key = "ASTRA_DB_APPLICATION_TOKEN", value = os.getenv("ASTRA_DB_APPLICATION_TOKEN"), disabled = True)
                #astra_db_endpoint = st.text_input("🌐 AstraDB API Endpoint:", type = "password", key = "ASTRA_DB_API_ENDPOINT", value = os.getenv("ASTRA_DB_API_ENDPOINT"), disabled = True)

                # Load API keys from .env file
                #groq_api_key = os.getenv("GROQ_API_KEY")
                #pinecone_api_key = os.getenv("PINECONE")
                #astra_db_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
                #astra_db_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")

                # Load API keys from Streamlit secrets
                groq_api_key = st.secrets.get("GROQ_API_KEY")
                pinecone_api_key = st.secrets.get("PINECONE")
                astra_db_token = st.secrets.get("ASTRA_DB_APPLICATION_TOKEN")
                astra_db_endpoint = st.secrets.get("ASTRA_DB_API_ENDPOINT")
                
                # Display masked status indicators (read-only)
                #st.text_input("🔑 Groq API Key:", value = "✅ Configured" if groq_api_key else "❌ Not Set", disabled = True, key = "GROQ_API_KEY")
                #st.text_input("🔑 Pinecone API Key:", value = "✅ Configured" if pinecone_api_key else "❌ Not Set", disabled = True, key = "PINECONE_API_KEY")
                #st.text_input("🔑 AstraDB Application Token:", value = "✅ Configured" if astra_db_token else "❌ Not Set", disabled = True, key = "ASTRA_DB_APPLICATION_TOKEN")
                #st.text_input("🌐 AstraDB API Endpoint:", value = "✅ Configured" if astra_db_endpoint else "❌ Not Set", disabled = True, key = "ASTRA_DB_API_ENDPOINT")

                selected_llm = st.selectbox("🧠 Select LLM Model", self.config.get_groq_model_options())

                temp_options = self.config.get_temperature()
                selected_temperature = st.slider("🔥 Temperature:", min_value = temp_options[0], max_value = temp_options[-1], value = temp_options[1])

                token_options = self.config.get_token()
                selected_token = st.slider("📏 Max Tokens:", min_value = token_options[0], max_value = token_options[-1], value = token_options[1])

                selected_vector_db = st.selectbox("🗄️ Select Vector Database", self.config.get_vector_db_options())

                uploaded_files = st.file_uploader(
                    "📂 Upload documents",
                    type=["pdf", "docx", "pptx", "csv", "txt", "xlsx"],
                    accept_multiple_files = True,
                    help = "Supported: .pdf, .docx, .pptx, .csv, .txt, .xlsx"
                )

                # Auto re-index when files or vector DB change
                if uploaded_files:
                    if self._needs_reindexing(uploaded_files, selected_vector_db):
                        self._process_and_index(
                            uploaded_files = uploaded_files,
                            selected_vector_db = selected_vector_db,
                            pinecone_api_key = pinecone_api_key,
                            astra_db_token = astra_db_token,
                            astra_db_endpoint = astra_db_endpoint
                        )

            # Chat Interface
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            # Display chat history
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            user_query = st.chat_input("Upload the file and then ask a question about your document")

            if user_query:
                logging.info(f"User query received: {user_query}")

                if "vector_store" not in st.session_state:
                    st.warning("⚠️ Please upload documents first.")
                    return

                if not groq_api_key:
                    st.warning("⚠️ Please enter your GROQ API key.")
                    return

                st.session_state["messages"].append({"role": "user", "content": user_query})

                # Display the user message immediately before generating response
                with st.chat_message("user"):
                    st.markdown(user_query)

                # Initialize response before the block to avoid UnboundLocalError
                response = None

                with st.chat_message("assistant"):
                    with st.spinner("🔍 Searching & generating answer..."):
                        try:
                            retriever_obj = Retriever(vector_store = st.session_state["vector_store"], top_k = self.config.get_top_k())

                            qa_chain = QAChain(
                                retriever = retriever_obj.get_retriever(),
                                groq_api_key = groq_api_key,
                                model_name = selected_llm,
                                temperature = selected_temperature,
                                max_tokens = int(selected_token)
                            )

                            response = qa_chain.run(user_query)
                            st.markdown(response)

                            logging.info("Response generated successfully.")

                        except Exception as e:
                            logging.exception("QA chain execution failed.")
                            response = f"❌ Error generating response: {e}"
                            st.error(response)

                if response is not None:
                    st.session_state["messages"].append({"role": "assistant", "content": response})

        except Exception as e:
            logging.exception("Fatal error while rendering Streamlit UI.")
            raise CustomException(e, sys)
