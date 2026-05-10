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
import streamlit as st

class TavilySearchTool:
    """
    Wrapper around Tavily Search API.

    Responsibilities:
        - Execute search queries
        - Normalize results into SearchResult format
        - Provide clean interface for SearchAgent
    """
    MAX_RESULTS = 3
    MAX_SNIPPET_CHARS = 500

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
            
            self._client = TavilySearchResults(
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

    def _clean_text(self, text: str) -> str:
        """
        Clean and truncate text.
        """

        if not text:
            return ""

        text = text.strip().replace("\n", " ")

        # Hard token safety truncation
        return text[:self.MAX_SNIPPET_CHARS]

    def search(self, query: str, max_results: int = MAX_RESULTS) -> List[SearchResult]:
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
            st.write(f"Search query: {query}")
            #response = self._client.invoke(query = query, max_results = max_results)
            query = query.strip()[:200]
            response = self._client.invoke(query)
            # ✅ Debug (remove later)
            st.write("📦 Response Type:", type(response))
            st.write("📦 Raw Response:", response)
            st.write(f"Tavily raw response: {response}")
            if not isinstance(response, list):
                logging.warning(f"Unexpected Tavily response: {type(response)}")
                return []
            
            # Normalize Results
            results: List[SearchResult] = []
            seen_urls = set()

            for item in response[:max_results]:
                url = item.get("url", "").strip()

                # Remove duplicates
                if not url or url in seen_urls:
                    continue
            
                seen_urls.add(url)

                # IMPORTANT:
                # Use SHORT snippets only
                snippet = self._clean_text(
                    item.get("content", "")
                )

                result: SearchResult = {
                    "url": url,
                    "title": self._clean_text(
                        item.get("title", "")
                    ),
                    "snippet": snippet,
                    "query": query,
                }
                results.append(result)

            logging.info("Tavily search completed | Query = '%s' | Results = %d", query, len(results))

            return results

        except Exception as e:
            logging.exception("Error during Tavily search | Query='%s'", query)
            raise CustomException(e, sys)