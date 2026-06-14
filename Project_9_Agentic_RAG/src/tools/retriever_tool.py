# Standard library imports
import sys
import streamlit as st
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_core.tools import tool


class RetrieverTool:
    """
    Wrapper class that creates a LangChain tool from the
    VectorStoreRetriever stored in Streamlit session state.

    Responsibilities:
    - Provide a proper LangChain @tool for the Agentic RAG pipeline
    - Retrieve relevant document chunks from the indexed vector store
    - Return formatted context from uploaded PDFs
    """

    def __init__(self, vector_store=None, top_k: int = 5):
        """
        Initialize RetrieverTool.

        Args:
            vector_store: Optional VectorStore instance (used during ingestion).
            top_k (int): Number of top similar documents to retrieve.
        """
        try:
            logging.info(f"Initializing RetrieverTool with top_k = {top_k}.")
            self.vector_store = vector_store
            self.top_k = top_k
            self._retriever = None

        except Exception as e:
            logging.exception("Error during RetrieverTool initialization.")
            raise CustomException(e, sys)

    def get_retriever(self):
        """
        Create and return a VectorStoreRetriever instance.
        Used during ingestion pipeline to store in session state.

        Returns:
            VectorStoreRetriever: Configured retriever instance.
        """
        try:
            if self._retriever is None:
                logging.info(
                    f"Creating VectorStoreRetriever with similarity search (top_k = {self.top_k})."
                )
                self._retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": self.top_k}
                )
                logging.info("VectorStoreRetriever created successfully.")
            return self._retriever

        except Exception as e:
            logging.exception("Error while creating VectorStoreRetriever.")
            raise CustomException(e, sys)

    @staticmethod
    def get_tool():
        """
        Returns a proper LangChain @tool for use in ToolNode / bind_tools.

        The tool retrieves documents from the session state vector retriever.
        If no documents are indexed yet, it returns a helpful message.
        """

        @tool
        def vector_db_retriever(query: str) -> str:
            """
            Search and retrieve information from user-uploaded documents.

            Use this tool when the user asks about:
            - Uploaded PDF documents
            - Document summaries
            - Information contained in uploaded files
            - Internal or proprietary knowledge
            - Questions that should be answered from the user's documents

            Prefer this tool whenever relevant uploaded documents are available.

            Do NOT use this tool for:
            - General knowledge questions
            - Current events or recent information
            - Academic research papers
            - Information that is not contained in the uploaded documents
            """
            try:
                retriever = st.session_state.get("vector_retriever")

                logging.info(f"vector_db_retriever tool invoked with query: {query}")
                docs = retriever.invoke(query)

                if not docs:
                    return "No relevant documents found for the given query."

                # Format documents into a single string for the LLM
                formatted = "\n\n".join(
                    f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
                    for doc in docs
                )
                return formatted

            except Exception as e:
                logging.exception("Error in vector_db_retriever tool.")
                return f"Error retrieving documents: {str(e)}"

        return vector_db_retriever