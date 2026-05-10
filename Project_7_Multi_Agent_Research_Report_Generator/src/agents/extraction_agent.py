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

Return ONLY valid JSON in this format (no commentary, no markdown fences):

{{
  "key_points": ["detailed point 1", "detailed point 2", "..."],
  "statistics": ["stat with number/year/source 1", "..."],
  "source_urls": ["url1", "url2", "..."]
}}

Rules:
- Provide 8 to 10 key_points (each 1-2 sentences, specific & informative).
- Provide 5 to 10 statistics with concrete numbers, percentages, dates,
  or named sources whenever possible.
- Include up to 10 source_urls actually present in the text.
- Do NOT invent facts. Use only the provided research text.
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

            # Keep text generous but bounded
            research_text = research_text[:10000]

            # Run LLM
            response = self.run(research_text)

            # Default fallback
            extracted_data = {
                "key_points": [],
                "statistics": [],
                "source_urls": []
            }

            try:
                # Strip possible markdown code fences
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    # remove leading "json" tag if present
                    if cleaned.lower().startswith("json"):
                        cleaned = cleaned[4:]
                parsed = json.loads(cleaned)

                extracted_data["key_points"] = parsed.get(
                    "key_points",
                    []
                )[:10]

                extracted_data["statistics"] = parsed.get(
                    "statistics",
                    []
                )[:10]

                extracted_data["source_urls"] = parsed.get(
                    "source_urls",
                    []
                )[:10]

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