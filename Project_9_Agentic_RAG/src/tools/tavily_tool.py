import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import warnings
warnings.filterwarnings("ignore")

class TavilySearchTool:
    """
    Wrapper around Tavily Search API.

    Responsibilities:
        - Search the web for current information.
        - Retrieve publicly available online content.
        - Expose a LangChain-compatible tool for Agentic RAG.
    """

    MAX_RESULTS = 3

    def __init__(self) -> None:
        """
        Initialize Tavily search client.

        Raises:
            CustomException: If Tavily initialization fails.
        """
        try:
            logging.info("Initializing TavilySearchTool.")

            tavily_api_key = st.secrets.get("TAVILY_API_KEY")

            if not tavily_api_key:
                raise ValueError( "TAVILY_API_KEY is missing.")

            self._tavily_tool = TavilySearchResults(
                tavily_api_key = tavily_api_key,
                max_results = self.MAX_RESULTS,
                search_depth = "advanced",
                include_answer = False,
                include_raw_content = False
            )

            logging.info("TavilySearchTool initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize TavilySearchTool.")
            raise CustomException(e, sys)

    def get_tool(self):
        """
        Create and return the Tavily search tool.

        Returns:
            Tool: LangChain-compatible tool instance.
        """

        tavily = self._tavily_tool

        @tool
        def tavily_web_search(query: str) -> str:
            """
            Search the web for publicly available information.

            Use this tool when the user asks about:
            - Current events
            - Recent news
            - Latest updates
            - Time-sensitive information
            - Information that changes frequently
            - Public web content

            Do NOT use this tool for:
            - User-uploaded documents
            - Internal knowledge bases
            - Academic research papers
            - General encyclopedia-style knowledge
            """
            try:
                logging.info("tavily_web_search invoked | Query = '%s'", query)

                results = tavily.invoke({"query": query})

                if not results:
                    logging.warning("No Tavily results found for query = '%s'", query)
                    return ("No relevant web search results were found." )

                formatted_results = []

                for result in results:
                    url = result.get("url", "N/A")
                    content = result.get("content", "" )

                    formatted_results.append(
                        f"Source: {url}\n"
                        f"Content: {content}"
                    )

                logging.info("Tavily search completed successfully.Results retrieved: %s", len(results))

                return "\n\n".join(formatted_results)

            except Exception as e:
                logging.exception("Error while executing tavily_web_search.")
                return (f"Error searching the web: {str(e)}")

        return tavily_web_search