import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from langchain_core.tools import tool
import warnings
warnings.filterwarnings("ignore")

class RetrieverTool:
    """
    Retriever manager for Agentic RAG.

    Responsibilities:
        - Create a VectorStoreRetriever.
        - Retrieve relevant content from indexed documents.
        - Expose a LangChain tool for tool-calling agents.
    """

    def __init__(self, vector_store = None, top_k: int = 5) -> None:
        """
        Initialize RetrieverTool.

        Args:
            vector_store: Vector database instance.
            top_k (int): Number of documents to retrieve.
        """
        try:
            logging.info("Initializing RetrieverTool | top_k = %s", top_k)

            self.vector_store = vector_store
            self.top_k = top_k
            self._retriever = None

        except Exception as e:
            logging.exception("Failed to initialize RetrieverTool.")
            raise CustomException(e, sys)

    def get_retriever(self):
        """
        Create and return a VectorStoreRetriever.

        Returns:
            VectorStoreRetriever
        """
        try:
            if self._retriever is None:

                logging.info("Creating VectorStoreRetriever | ""search_type=similarity | top_k = %s", self.top_k)

                self._retriever = self.vector_store.as_retriever(
                    search_type = "similarity",
                    search_kwargs = {"k": self.top_k}
                )

                logging.info("VectorStoreRetriever created successfully.")

            return self._retriever

        except Exception as e:
            logging.exception("Failed to create VectorStoreRetriever.")
            raise CustomException(e, sys)

    @staticmethod
    def get_tool():
        """
        Create LangChain retriever tool.

        Returns:
            Tool: LangChain-compatible tool.
        """

        @tool
        def vector_db_retriever(query: str) -> str:
            """
            Search and retrieve information from user-uploaded documents.

            Use this tool when the user asks about:
            - Uploaded PDF documents
            - Document summaries
            - Information contained in uploaded files
            - Internal knowledge
            - Proprietary content
            - Questions that reference uploaded documents

            Prefer this tool whenever indexed documents
            may contain the answer.

            Do NOT use this tool for:
            - General knowledge
            - Current events
            - Breaking news
            - Academic paper search
            - Information outside uploaded documents
            """
            try:
                retriever = st.session_state.get("vector_retriever")

                if retriever is None:

                    logging.warning("vector_db_retriever called but no retriever exists in session.")

                    return ("No documents have been uploaded or indexed in the current session.")

                logging.info("vector_db_retriever invoked | Query = '%s'", query)

                docs = retriever.invoke(query)

                if not docs:

                    logging.info("No matching documents found for query = '%s'", query)

                    return ("No relevant information was found in the uploaded documents." )

                formatted_docs = []

                for doc in docs:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A" )

                    formatted_docs.append(
                        f"[Source: {source} | Page: {page}]\n"
                        f"{doc.page_content}"
                    )

                logging.info("Retrieved %s document chunks.", len(docs))

                return "\n\n".join(formatted_docs)

            except Exception as e:
                logging.exception("Error while executing vector_db_retriever.")
                return (f"Document retrieval failed: {str(e)}")

        return vector_db_retriever