# Standard Library Imports
import sys
import warnings
from typing import Any, Dict
from langchain_groq import ChatGroq
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.agents.base_agent import BaseAgent
from src.models.state import ResearchState, ExtractedData
warnings.filterwarnings("ignore")

# Writer Agent Prompt
_WRITER_PROMPT: str = """
You are a Professional Research Report Writer.

Using the provided structured data, generate a high-quality research report.

Your report MUST include:
1. Title
2. Executive Summary
3. Introduction
4. Main Sections (based on themes)
5. Key Statistics (highlighted)
6. Notable Insights / Quotes
7. Conclusion
8. References (use source URLs)

Guidelines:
- Use clear headings (Markdown format)
- Expand each theme into a well-written section
- Keep it informative and professional
- Do NOT hallucinate facts
- Use only provided data
"""

class WriterAgent(BaseAgent):
    """
    Writer Agent

    Responsibilities:
        - Convert structured extracted data into a full research report

    Input (from ResearchState):
        - extracted_data (ExtractedData)

    Output (to ResearchState):
        - draft_report (str)
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize WriterAgent with system prompt.

        Args:
            llm (ChatGroq): Shared LLM instance

        Raises:
            CustomException: If initialization fails
        """
                
        try:
            logging.info("Initializing WriterAgent...")

            super().__init__(llm, _WRITER_PROMPT)

            logging.info("WriterAgent initialized successfully.")

        except Exception as e:
            logging.exception("Error initializing WriterAgent initialization.")
            raise CustomException(e, sys)

    def execute(self, state: ResearchState) -> Dict[str, Any]:
        """
        Generate research report from extracted data.

        Args:
            state (ResearchState): Workflow state

        Returns:
            Dict[str, Any]:
                {
                    "draft_report": str
                }

        Raises:
            CustomException: If execution fails
        """

        try:
            logging.info("WRITER AGENT EXECUTION  START")

            # Validate input
            extracted_data: ExtractedData = state.get("extracted_data")

            if not extracted_data:
                raise ValueError("Extracted data not found in state.")

            logging.info("Extracted data received successfully.")
            
            # Prepare structured input for LLM
            themes = extracted_data.get("themes", [])
            key_stats = extracted_data.get("key_stats", [])
            quotes = extracted_data.get("notable_quotes", [])
            sources = extracted_data.get("source_urls", [])

            logging.info(
                "Input Summary | Themes = %d | Stats = %d | Quotes = %d | Sources = %d",
                len(themes), len(key_stats), len(quotes), len(sources)
            )

            # Convert structured data into readable prompt text
            input_text = f"""
                Themes:{themes}
                Key Statistics:{key_stats}
                Notable Quotes:{quotes}
                Sources:{sources}
            """
            logging.debug("Prepared input text for LLM (length=%d chars).", len(input_text))

            # Run LLM
            report: str = self.run(input_text)

            # Validate output
            if not report or not report.strip():
                logging.warning("Empty report generated.")

            logging.info("Report generated successfully | Length = %d", len(report) if report else 0)

            logging.info("WRITER AGENT EXECUTION  END")

            return {"draft_report": report}

        except Exception as e:
            logging.exception("Error during WriterAgent execution.")
            raise CustomException(e, sys)