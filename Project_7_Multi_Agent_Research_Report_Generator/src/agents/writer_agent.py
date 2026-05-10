import sys
import warnings
from typing import Any, Dict

from langchain_groq import ChatGroq

from src.agents.base_agent import BaseAgent
from src.models.state import ResearchState
from src.utils.logger import logging
from src.utils.exception import CustomException

warnings.filterwarnings("ignore")


_WRITER_PROMPT = """
You are a Research Report Writer.

Generate a SHORT professional report using the provided data.

Rules:
- Keep report concise
- Use markdown headings
- Use only provided information
- Avoid repetition
- Keep report under 1000 words
"""


class WriterAgent(BaseAgent):

    def __init__(self, llm: ChatGroq) -> None:

        try:
            logging.info("Initializing WriterAgent")

            super().__init__(llm, _WRITER_PROMPT)

            logging.info("WriterAgent initialized.")

        except Exception as e:
            logging.exception("WriterAgent initialization failed.")
            raise CustomException(e, sys)

    def execute(self, state: ResearchState) -> Dict[str, Any]:

        try:
            logging.info("WRITER AGENT START")

            extracted_data = state.get(
                "extracted_data",
                {}
            )

            if not extracted_data:
                raise ValueError(
                    "No extracted data found."
                )

            # Keep only limited data
            key_points = extracted_data.get(
                "key_points",
                []
            )[:5]

            statistics = extracted_data.get(
                "statistics",
                []
            )[:5]

            sources = extracted_data.get(
                "source_urls",
                []
            )[:5]

            # Compact input
            input_text = f"""
            Key Points:
            {key_points}

            Statistics:
            {statistics}

            Sources:
            {sources}
            """

            # Prevent token overflow
            input_text = input_text[:4000]

            # Generate report
            report = self.run(input_text)

            if not report:
                report = "Report generation failed."

            # Final hard limit
            report = report[:12000]

            logging.info(
                "Report generated successfully."
            )

            logging.info("WRITER AGENT END")

            return {
                "draft_report": report
            }

        except Exception as e:
            logging.exception(
                "Error during WriterAgent execution."
            )

            raise CustomException(e, sys)