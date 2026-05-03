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
_SEARCH_PROMPT: str = """
You are a search specialist. Given search results, summarize key findings 
in 1–2 sentences per query. Be factual and concise.
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

            all_results: List[SearchResult] = []
            
            # Execute search per query
            for query in queries:
                try:
                    clean_query = query.strip()

                    if not clean_query:
                        continue

                    logging.info("Searching for query: %s", clean_query)

                    results = self._search_tool.search(clean_query)

                    if not results:
                        logging.warning("No results found for query: %s", clean_query)
                        continue
                        
                    logging.info("Query completed | Results = %d", len(results))

                    all_results.extend(results)

                except Exception:
                    logging.exception("Search failed for query: %s (continuing)", query)
                    continue
          
            # Final validation
            if not all_results:
                logging.warning("No search results collected from any query.")
            
            logging.info("Total aggregated results: %d", len(all_results))

            logging.info("SEARCH AGENT END")

            return {"raw_search_results": all_results}

        except Exception as e:
            logging.exception("Error during SearchAgent execution.")
            raise CustomException(e, sys)