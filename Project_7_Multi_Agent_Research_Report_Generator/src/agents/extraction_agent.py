import sys
import json
import warnings
from typing import Any, Dict

from langchain_groq import ChatGroq

from src.agents.base_agent import BaseAgent
from src.models.state import ResearchState
from src.utils.logger import logging
from src.utils.exception import CustomException

warnings.filterwarnings("ignore")


_EXTRACTION_PROMPT = """
Extract important insights from the research text.

Return ONLY valid JSON in this format:

{
  "key_points": ["point 1", "point 2"],
  "statistics": ["stat 1", "stat 2"],
  "source_urls": ["url1", "url2"]
}
"""


class ExtractionAgent(BaseAgent):

    def __init__(self, llm: ChatGroq) -> None:

        try:
            logging.info("Initializing ExtractionAgent")

            super().__init__(llm, _EXTRACTION_PROMPT)

            logging.info("ExtractionAgent initialized.")

        except Exception as e:
            logging.exception("Error initializing ExtractionAgent")
            raise CustomException(e, sys)

    def execute(self, state: ResearchState) -> Dict[str, Any]:

        try:
            logging.info("EXTRACTION AGENT START")

            # Use compressed research instead of raw results
            research_text = state.get(
                "compressed_research",
                ""
            )

            if not research_text:
                raise ValueError("No research text found.")

            # Keep text short
            research_text = research_text[:4000]

            # Run LLM
            response = self.run(research_text)

            # Default fallback
            extracted_data = {
                "key_points": [],
                "statistics": [],
                "source_urls": []
            }

            try:
                parsed = json.loads(response)

                extracted_data["key_points"] = parsed.get(
                    "key_points",
                    []
                )[:5]

                extracted_data["statistics"] = parsed.get(
                    "statistics",
                    []
                )[:5]

                extracted_data["source_urls"] = parsed.get(
                    "source_urls",
                    []
                )[:5]

            except Exception:
                logging.warning(
                    "JSON parsing failed. Using fallback."
                )

            logging.info(
                "Extraction completed."
            )

            logging.info("EXTRACTION AGENT END")

            return {
                "extracted_data": extracted_data
            }

        except Exception as e:
            logging.exception(
                "Error during ExtractionAgent execution."
            )

            raise CustomException(e, sys)