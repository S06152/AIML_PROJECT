import sys
import json
import warnings
from typing import Any, Dict, List
from langchain_groq import ChatGroq
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.agents.base_agent import BaseAgent
from src.models.state import ResearchState, ExtractedData, SearchResult
warnings.filterwarnings("ignore")

# Extraction Agent Prompt
_EXTRACTION_PROMPT: str = """
You are a Research Extraction Agent.

Your task is to analyze search results and extract structured insights.

From the given content, extract:

1. Key Themes (with short descriptions)
2. Important Statistics or Data Points
3. Notable Quotes or Insights
4. Source URLs

Output strictly in JSON format:

{
  "themes": [
    {"theme": "...", "description": "..."}
  ],
  "key_stats": ["..."],
  "notable_quotes": ["..."],
  "source_urls": ["..."]
}
"""

class ExtractionAgent(BaseAgent):
    """
    Extraction Agent

    Responsibilities:
        - Process raw search results
        - Extract structured insights using LLM
        - Return structured ExtractedData

    Input (from ResearchState):
        - raw_search_results (List[SearchResult])

    Output (to ResearchState):
        - extracted_data (ExtractedData)
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize ExtractionAgent with system prompt.

        Args:
            llm (ChatGroq): Shared LLM instance

        Raises:
            CustomException: If initialization fails
        """
                
        try:
            logging.info("Initializing ExtractionAgent...")

            super().__init__(llm, _EXTRACTION_PROMPT)

            logging.info("ExtractionAgent initialized successfully.")

        except Exception as e:
            logging.exception("Error initializing ExtractionAgent.")
            raise CustomException(e, sys)

    def execute(self, state: ResearchState) -> Dict[str, Any]:
        """
        Extract structured insights from raw search results.

        Args:
            state (ResearchState): Workflow state containing search results

        Returns:
            Dict[str, Any]:
                {
                    "extracted_data": ExtractedData
                }

        Raises:
            CustomException: If execution fails
        """
        try:
            logging.info("EXTRACTION AGENT START")

            # Validate input
            results: List[SearchResult] = state.get("raw_search_results", [])

            if not results:
                raise ValueError("raw_search_results is empty.")

            logging.info("Processing %d search results.", len(results))

            # Prepare input text for LLM (limit to avoid token overflow)
            combined_text_parts: List[str] = []

            for idx, result in enumerate(results[:10]):  # limit to top 10
                combined_text_parts.append(
                    f"""
                        Result: {idx + 1}
                        Title: {result.get('title', '')}
                        Snippet: {result.get('snippet', '')}
                        URL: {result.get('url', '')}
                    """
                )

            combined_text: str = "\n".join(combined_text_parts)
            logging.debug("Prepared combined text for LLM (length = %d chars).", len(combined_text))

           # Run LLM extraction
            response: str = self.run(combined_text)

            logging.info("LLM extraction response received.")

            # Parse JSON output
            try:
                extracted: ExtractedData = json.loads(response)
                
                # Basic schema validation
                if not isinstance(extracted, dict):
                    raise ValueError("Parsed JSON is not a dictionary.")

                extracted.setdefault("themes", [])
                extracted.setdefault("key_stats", [])
                extracted.setdefault("notable_quotes", [])
                extracted.setdefault("source_urls", [])

                logging.info("Extracted data parsed successfully.")

            except Exception as parse_error:
                logging.warning("JSON parsing failed. Falling back to empty structure. Error: %s", str(parse_error))

                extracted = {
                    "themes": [],
                    "key_stats": [],
                    "notable_quotes": [],
                    "source_urls": []
                }
            
            logging.info(
                "Extraction summary | Themes = %d | Stats = %d | Quotes = %d | Sources = %d",
                len(extracted.get("themes", [])),
                len(extracted.get("key_stats", [])),
                len(extracted.get("notable_quotes", [])),
                len(extracted.get("source_urls", [])),
            )

            logging.info("EXTRACTION AGENT EXECUTION  END")

            return {"extracted_data": extracted}

        except Exception as e:
            logging.exception("Error during ExtractionAgent execution.")
            raise CustomException(e, sys)