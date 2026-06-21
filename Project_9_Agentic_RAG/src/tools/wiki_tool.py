import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
import warnings
warnings.filterwarnings("ignore")

class WikiTool:
    """
    Wrapper around WikipediaAPIWrapper.

    Responsibilities:
        - Search Wikipedia for general knowledge.
        - Retrieve encyclopedic information.
        - Expose a LangChain-compatible tool for Agentic RAG.
    """

    def __init__(self) -> None:
        """
        Initialize Wikipedia API wrapper.

        Raises:
            CustomException: If initialization fails.
        """
        try:
            self._api_wrapper = WikipediaAPIWrapper(top_k_results = 2, doc_content_chars_max = 3000)

            logging.info("WikiTool initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize WikiTool.")
            raise CustomException(e, sys)

    def get_tool(self):
        """
        Create and return the Wikipedia search tool.

        Returns:
            Tool: LangChain-compatible tool instance.
        """

        api_wrapper = self._api_wrapper

        @tool
        def wikipedia_search(query: str) -> str:
            """
            Search Wikipedia for general knowledge,
            factual information, definitions, concepts,
            historical events, people, places, organizations,
            and background information.

            Use this tool when the user asks about:
            - General knowledge
            - Definitions and explanations
            - Historical events
            - Famous people
            - Countries, cities, organizations
            - Scientific concepts
            - Encyclopedia-style information

            Do NOT use this tool for:
            - Current events
            - Breaking news
            - Real-time information
            - Academic research papers
            - User-uploaded documents
            """
            try:
                logging.info("wikipedia_search invoked | Query = '%s'", query)

                result = api_wrapper.run(query)

                if not result:
                    logging.warning("No Wikipedia results found for query = '%s'", query )
                    return ("No relevant Wikipedia information was found for the given query.")

                logging.info("Wikipedia search completed successfully.")

                return result

            except Exception as e:
                logging.exception("Error while executing wikipedia_search.")
                return (f"Error searching Wikipedia: {str(e)}")

        return wikipedia_search