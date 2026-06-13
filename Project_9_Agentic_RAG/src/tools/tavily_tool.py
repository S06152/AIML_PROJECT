# Standard Library Imports
import sys
import warnings
import streamlit as st
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
warnings.filterwarnings("ignore")

class TavilySearchTool:
    """
    Wrapper around Tavily Search API.

    Responsibilities:
        - Execute search queries
        - Normalize results into SearchResult format
        - Provide clean interface for SearchAgent
    """
    MAX_RESULTS = 3

    def __init__(self) -> None:
        """
        Initialize Tavily search client.

        Args:
            api_key (str): Tavily API Key

        Raises:
            CustomException: If initialization fails
        """
                
        try:
            logging.info("Initializing TavilySearchTool...")

            tavily_api_key = st.secrets.get("TAVILY_API_KEY")

            if not tavily_api_key:
                raise ValueError("TAVILY_API_KEY missing.")  
            
            self._tavily_tool = TavilySearchResults(
                tavily_api_key = tavily_api_key,
                max_results = self.MAX_RESULTS,
                search_depth = "basic",
                include_answer = False,
                include_raw_content = False
            )

            logging.info("TavilySearchTool initialized successfully.")

        except Exception as e:
            logging.exception("Error initializing TavilySearchTool.")
            raise CustomException(e, sys)

    def get_tool(self):
        tavily = self._tavily_tool

        @tool
        def tavily_web_search(query: str) -> str:
            """Search the web for current events, recent news, live information, 
            or any real-time data. Use this tool when the user asks about recent 
            happenings, latest updates, current prices, weather, or anything that 
            requires up-to-date web information."""
            try:
                logging.info(f"tavily_web_search tool invoked with query: {query}")
                results = tavily.invoke({"query": query})
                if not results:
                    return "No web search results found for the given query."
                # Format results into readable text
                formatted = "\n\n".join(
                    f"Source: {r.get('url', 'N/A')}\n{r.get('content', '')}"
                    for r in results
                )
                return formatted
            except Exception as e:
                logging.exception("Error in tavily_web_search tool.")
                return f"Error searching the web: {str(e)}"

        return tavily_web_search