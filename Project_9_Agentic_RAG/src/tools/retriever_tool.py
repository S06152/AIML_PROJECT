import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
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

        NOTE: This tool previously read the retriever from
        `st.session_state.get("vector_retriever")`. `st.session_state` is
        only valid inside an active Streamlit ScriptRunContext. When the
        LangGraph workflow is executed from inside the FastAPI process
        (as this project does), there is no such context, so that lookup
        was invalid outside of the original single-process demo.

        This version instead reads the retriever via
        `RetrieverTool.get_active_retriever()`, which the FastAPI `/query`
        endpoint sets for the current request (using the session-scoped
        retriever resolved from its own SESSION_STORE). The retriever
        object itself is intentionally kept OUT of LangGraph state/
        `user_controls`, since the graph is now compiled with a
        checkpointer that serializes state on every turn -- a live,
        non-serializable retriever object would break that.
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
                retriever = RetrieverTool.get_active_retriever()

                if retriever is None:

                    logging.warning("vector_db_retriever called but no retriever is available for this request.")

                    return ("No documents have been uploaded or indexed in the current session.")

                logging.info("vector_db_retriever invoked | Query = \'%s\'", query)

                docs = retriever.invoke(query)

                if not docs:

                    logging.info("No matching documents found for query = \'%s\'", query)

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

    # ------------------------------------------------------------------
    # Request-scoped retriever access (replaces st.session_state usage)
    # ------------------------------------------------------------------
    #
    # `agents.py` sets this immediately before invoking the LLM for a
    # given request, using the retriever resolved by the FastAPI layer
    # for that request's session_id. This avoids relying on Streamlit
    # session state inside the FastAPI process, while still letting the
    # `@tool`-decorated function above (which LangGraph calls with only
    # `query` as an argument) access the correct retriever.
    #
    # This is a pragmatic, low-risk fix. It is still process-global
    # (not safe across concurrent requests handled by separate workers);
    # see the production-readiness notes for a fuller multi-tenant fix
    # (e.g. injecting retrievers via LangGraph's InjectedState).
    _active_retriever = None

    @classmethod
    def set_active_retriever(cls, retriever) -> None:
        cls._active_retriever = retriever

    @classmethod
    def get_active_retriever(cls):
        return cls._active_retriever
