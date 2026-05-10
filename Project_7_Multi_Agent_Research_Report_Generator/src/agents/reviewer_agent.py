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

_REVIEWER_PROMPT = """
Review the research report.

Return ONLY valid JSON:

{{
  "approved": true,
  "overall_score": 8,
  "suggestions": ["suggestion 1"]
}}

Rules:
- Approve if report is reasonably good
- Keep suggestions short
"""

class ReviewerAgent(BaseAgent):
    MAX_REVISIONS = 1

    def __init__(self, llm: ChatGroq) -> None:

        try:
            logging.info("Initializing ReviewerAgent")

            super().__init__(llm, _REVIEWER_PROMPT)

            logging.info("ReviewerAgent initialized.")

        except Exception as e:
            logging.exception("ReviewerAgent initialization failed.")
            raise CustomException(e, sys)

    def execute(self, state: ResearchState) -> Dict[str, Any]:

        try:
            logging.info("REVIEWER AGENT START")

            report = state.get("draft_report","")
            revision_count = state.get("revision_count", 0)

            if not report:
                raise ValueError("Draft report missing.")

            # Keep report short for review
            review_text = report[:5000]

            # Run review
            response = self.run(review_text)

            # Default feedback
            feedback = {
                "approved": True,
                "overall_score": 7,
                "suggestions": []
            }

            try:
                parsed = json.loads(response)

                feedback["approved"] = parsed.get("approved", True)
                feedback["overall_score"] = parsed.get("overall_score", 7)
                feedback["suggestions"] = parsed.get("suggestions", [])[:3]

            except Exception:
                logging.warning("Failed to parse review response.")

            # Final decision
            if (feedback["approved"]or revision_count >= self.MAX_REVISIONS):
                logging.info("Finalizing report.")

                return {
                    "review_feedback": feedback,
                    "final_report": report,
                    "revision_count": revision_count
                }

            # Request rewrite
            logging.info("Revision required.")

            return {
                "review_feedback": feedback,
                "revision_count": revision_count + 1
            }

        except Exception as e:
            logging.exception("Error during ReviewerAgent execution.")
            raise CustomException(e, sys)