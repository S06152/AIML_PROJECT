# Standard Library Imports
import sys
import warnings
from typing import List
import streamlit as st
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.models.state import SearchResult
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
                raise ValueError("TAVILY_API_KEY not found in environment variables.")  

            self._client = TavilySearchResults(tavily_api_key = tavily_api_key)

            logging.info("TavilySearchTool initialized successfully.")

        except Exception as e:
            logging.exception("Error initializing TavilySearchTool.")
            raise CustomException(e, sys)

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Perform search using Tavily API.

        Args:
            query (str): Search query
            max_results (int): Maximum number of results

        Returns:
            List[SearchResult]: Normalized search results
        """

        try:
            if not query or not query.strip():
                raise ValueError("Search query cannot be empty.")
            
            logging.info("Tavily search started | Quer = '%s' | MaxResults = %d", query, max_results)

            response = self._client.invoke(query = query, max_results = max_results)

            if not isinstance(response, dict):
                logging.warning("Unexpected Tavily response format.")
                return []
            
            # Normalize Results
            results: List[SearchResult] = []

            for item in response.get("results", []):
                result: SearchResult = {
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "query": query,
                }
                results.append(result)

            logging.info("Tavily search completed | Query = '%s' | Results = %d", query, len(results))

            return results

        except Exception as e:
            logging.exception("Error during Tavily search | Query='%s'", query)
            raise CustomException(e, sys)