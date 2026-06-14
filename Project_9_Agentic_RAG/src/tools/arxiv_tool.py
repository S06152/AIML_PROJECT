import sys
import warnings

from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool

from src.utils.logger import logging
from src.utils.exception import CustomException

warnings.filterwarnings("ignore")


class ArxivTool:

    def __init__(self):
        try:
            self._api_wrapper = ArxivAPIWrapper(
                top_k_results=3,
                load_max_docs=3,
                doc_content_chars_max=4000
            )

            logging.info("Arxiv Tool initialized")

        except Exception as e:
            logging.exception("Failed to initialize Arxiv Tool.")
            raise CustomException(e, sys)

    def get_tool(self):
        api_wrapper = self._api_wrapper

        @tool
        def arxiv_search(query: str) -> str:
            """
            Search arXiv for academic and scientific literature.

            Use this tool when the user asks for:
            - Research papers
            - Academic publications
            - Scientific studies
            - Technical literature
            - Scholarly findings
            - State-of-the-art research

            Do not use this tool for:
            - Current news or real-time information
            - General encyclopedic knowledge
            - User-uploaded documents or private knowledge bases
            """
            try:
                logging.info(
                    f"arxiv_search tool invoked with query: {query}"
                )

                result = api_wrapper.run(query)

                return (
                    result
                    if result
                    else "No arXiv results found for the given query."
                )

            except Exception as e:
                logging.exception("Error in arxiv_search tool.")
                return f"Error searching arXiv: {str(e)}"

        return arxiv_search