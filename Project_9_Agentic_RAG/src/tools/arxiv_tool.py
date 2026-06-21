import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool
import warnings
warnings.filterwarnings("ignore")

class ArxivTool:
    """
    Wrapper around LangChain ArxivAPIWrapper.

    Responsibilities:
        - Search arXiv for academic papers.
        - Retrieve scientific and technical literature.
        - Expose a LangChain-compatible tool for Agentic RAG.
    """

    def __init__(self):
        """
        Initialize Arxiv API wrapper.

        Raises:
            CustomException: If initialization fails.
        """
        try:
            self._api_wrapper = ArxivAPIWrapper(top_k_results = 3, load_max_docs = 3, doc_content_chars_max = 4000)

            logging.info("ArxivTool initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize ArxivTool.")
            raise CustomException(e, sys)

    def get_tool(self):
        """
        Create and return the LangChain tool.

        Returns:
            Tool: LangChain-compatible tool instance.
        """

        api_wrapper = self._api_wrapper

        @tool
        def arxiv_search(query: str) -> str:
            """
            Search arXiv for academic research papers,
            scientific publications, technical reports,
            and scholarly literature.

            Use this tool when the user asks about:
            - Research papers
            - Academic publications
            - Scientific studies
            - Technical literature
            - Machine Learning research
            - AI papers
            - Physics, Mathematics, Computer Science
            - State-of-the-art research findings

            Do NOT use this tool for:
            - Current news
            - Real-time information
            - General encyclopedic knowledge
            - User-uploaded documents
            - Private knowledge bases
            """
            try:
                logging.info("arxiv_search tool invoked | Query='%s'", query)

                result = api_wrapper.run(query)

                if not result:
                    logging.warning("No arXiv results found for query='%s'", query)
                    return ("No arXiv research papers found for the given query.")

                logging.info("arXiv search completed successfully.")

                return result

            except Exception as e:
                logging.exception("Error while executing arxiv_search.")
                return (f"Error searching arXiv: {str(e)}")

        return arxiv_search