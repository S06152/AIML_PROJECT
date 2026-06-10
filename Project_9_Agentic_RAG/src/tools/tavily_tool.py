# Standard Library Imports
import sys
import warnings
import streamlit as st
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_community.tools.tavily_search import TavilySearchResults
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
            
            self._tool = TavilySearchResults(
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
        return self._tool