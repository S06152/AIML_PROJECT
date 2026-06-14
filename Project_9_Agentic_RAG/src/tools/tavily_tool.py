import sys
import warnings

import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from src.utils.logger import logging
from src.utils.exception import CustomException

warnings.filterwarnings("ignore")


class TavilySearchTool:
    """
    Wrapper around Tavily Search API.
    """

    MAX_RESULTS = 3

    def __init__(self) -> None:
        try:
            logging.info("Initializing TavilySearchTool...")

            tavily_api_key = st.secrets.get("TAVILY_API_KEY")

            if not tavily_api_key:
                raise ValueError("TAVILY_API_KEY is missing.")

            self._tavily_tool = TavilySearchResults(
                tavily_api_key=tavily_api_key,
                max_results=self.MAX_RESULTS,
                search_depth="advanced",  # use "basic" if you want fewer credits
                include_answer=False,
                include_raw_content=False,
            )

            logging.info("TavilySearchTool initialized successfully.")

        except Exception as e:
            logging.exception("Error initializing TavilySearchTool.")
            raise CustomException(e, sys)

    def get_tool(self):
        tavily = self._tavily_tool

        @tool
        def tavily_web_search(query: str) -> str:
            """
            Search the web for publicly available information.

            Use this tool when the user asks for:
            - Information available on websites
            - Recent or time-sensitive information
            - Current events, news, and updates
            - Information that may change over time
            - Information beyond the model's built-in knowledge

            Do NOT use this tool for:
            - User-uploaded documents or private knowledge bases
            - Academic research papers and scholarly literature
            - General encyclopedic background information when a knowledge source is more appropriate
            """
            try:
                logging.info(
                    f"tavily_web_search tool invoked with query: {query}"
                )

                results = tavily.invoke({"query": query})

                if not results:
                    return "No web search results found."

                formatted_results = []

                for result in results:
                    formatted_results.append(
                        f"Source: {result.get('url', 'N/A')}\n"
                        f"Content: {result.get('content', '')}"
                    )

                return "\n\n".join(formatted_results)

            except Exception as e:
                logging.exception("Error in tavily_web_search tool.")
                return f"Error searching the web: {str(e)}"

        return tavily_web_search