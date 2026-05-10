import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Any, Dict, List
from langchain_groq import ChatGroq
from src.agents.base_agent import BaseAgent
from src.models.state import ResearchState, SearchResult
from src.tools.search_tool import TavilySearchTool
import warnings
warnings.filterwarnings("ignore")

# Search Agent Prompt (Optional summarization)
_SEARCH_PROMPT = """
Summarize the search results in short bullet points.
Keep the response concise.
"""

class SearchAgent(BaseAgent):
    """
    Search Agent

    Responsibilities:
        - Execute web search using Tavily
        - Process multiple queries
        - Aggregate structured search results

    Input:
        state["search_queries"]

    Output:
        {
            "raw_search_results": List[SearchResult]
        }
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize SearchAgent with AUTOSAR-aware architect system prompt.

        Args:
            llm (ChatGroq): Shared Groq LLM instance from LLMProvider.
        """
        try:
            logging.info("Initializing SearchAgent")
            super().__init__(llm, _SEARCH_PROMPT)

            # Initialize Tavily search tool
            self._search_tool = TavilySearchTool()

            logging.info("SearchAgent initialized successfully.")

        except Exception as e:
            logging.exception("Error initializing SearchAgent.")
            raise CustomException(e, sys)

    def execute(self, state: ResearchState) -> Dict[str, Any]:
        """
        Execute search queries and collect results.

        Args:
            state (ResearchState)

        Returns:
            Dict:
                {
                    "raw_search_results": List[SearchResult]
                }
        """

        try:
            logging.info("SEARCH AGENT START")
            
            # Get queries safely
            queries: List[str] = state.get("search_queries", [])

            if not queries:
                logging.warning("No search queries found. Using fallback query.")
                topic = state.get("topic", "")
                queries = [topic] if topic else []
            
            if not queries:
                raise ValueError("No search queries found in state.")
        
            logging.info("Executing %d queries", len(queries))

            # Keep only 2 queries
            queries = queries[:2]

            all_results: List[SearchResult] = []
            
            compressed_research = ""

            for query in queries:

                query = query.strip()

                if not query:
                    continue

                logging.info("Searching: %s", query)

                results = self._search_tool.search(
                    query,
                    max_results=2
                )

                for item in results:

                    # Store lightweight result
                    result = {
                        "title": item.get("title", "")[:100],
                        "snippet": item.get("snippet", "")[:300],
                        "url": item.get("url", "")
                    }

                    all_results.append(result)

                    # Build compressed text
                    compressed_research += (
                        f"Title: {result['title']}\n"
                        f"Summary: {result['snippet']}\n\n"
                    )

            # Keep compressed research short
            compressed_research = compressed_research[:3000]

            logging.info(
                "Total Results=%d",
                len(all_results)
            )

            logging.info("SEARCH AGENT END")

            return {
                "compressed_research": compressed_research,
                "raw_search_results": all_results
            }

        except Exception as e:
            logging.exception("Error during SearchAgent execution.")
            raise CustomException(e, sys)