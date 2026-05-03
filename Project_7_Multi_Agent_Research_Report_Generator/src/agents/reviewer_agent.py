# Standard Library Imports
import sys
import json
import warnings
from typing import Any, Dict
from langchain_groq import ChatGroq
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.agents.base_agent import BaseAgent
from src.models.state import ResearchState, ReviewFeedback
warnings.filterwarnings("ignore")

# Reviewer Agent Prompt
_REVIEWER_PROMPT: str = """
You are a Senior Research Reviewer.

Your task is to critically evaluate the given research report.

Evaluate based on:
1. Clarity and readability
2. Completeness of coverage
3. Logical structure
4. Use of supporting data
5. Professional tone

Provide output STRICTLY in JSON format:

{
  "approved": true/false,
  "issues": ["..."],
  "suggestions": ["..."],
  "overall_score": 1-10
}

Rules:
- Approve ONLY if the report is high quality (score >= 8)
- Be critical and constructive
"""

class ReviewerAgent(BaseAgent):
    """
    Reviewer Agent

    Responsibilities:
        - Evaluate generated report quality
        - Provide structured feedback
        - Decide whether revision is needed

    Input (from ResearchState):
        - draft_report (str)
        - revision_count (int)

    Output (to ResearchState):
        - review_feedback (ReviewFeedback)
        - final_report (str) [if approved OR max revisions reached]
        - revision_count (int)
    """

    MAX_REVISIONS = 2

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize ReviewerAgent with system prompt.

        Args:
            llm (ChatGroq): Shared LLM instance

        Raises:
            CustomException: If initialization fails
        """
                
        try:
            logging.info("Initializing ReviewerAgent...")

            super().__init__(llm, _REVIEWER_PROMPT)

            logging.info("ReviewerAgent initialized successfully.")

        except Exception as e:
            logging.exception("Error initializing ReviewerAgent.")
            raise CustomException(e, sys)

    def execute(self, state: ResearchState) -> Dict[str, Any]:
        """
        Review the generated report and decide next step.

        Args:
            state (ResearchState): Workflow state

        Returns:
            Dict[str, Any]:
                {
                    "review_feedback": ReviewFeedback,
                    "final_report": str (optional),
                    "revision_count": int
                }

        Raises:
            CustomException: If execution fails
        """

        try:
            logging.info("REVIEWER AGENT START")

            # Validate input
            report: str = state.get("draft_report", "")
            revision_count: int = state.get("revision_count", 0)

            if not report or not report.strip():
                raise ValueError("Draft report not found or empty")

            logging.info("Reviewing report | Length = %d | RevisionCount = %d", len(report), revision_count)
           
            # Run LLM review
            response: str = self.run(report)

            logging.info("LLM review response received.")

            # Parse JSON response
            try:
                feedback: ReviewFeedback = json.loads(response)

                # Basic validation
                if not isinstance(feedback, dict):
                    raise ValueError("Parsed feedback is not a dictionary.")

                feedback.setdefault("approved", False)
                feedback.setdefault("issues", [])
                feedback.setdefault("suggestions", [])
                feedback.setdefault("overall_score", 0)

                logging.info("Review feedback parsed successfully.")

            except Exception as parse_error:
                logging.warning("Failed to parse review JSON. Using fallback. Error: %s", str(parse_error))

                feedback = {
                    "approved": False,
                    "issues": ["Failed to parse LLM response"],
                    "suggestions": ["Retry review or improve prompt formatting"],
                    "overall_score": 5
                }

            approved: bool = feedback.get("approved", False)
            score: int = feedback.get("overall_score", 0)

            logging.info("Review Result | Approved=%s | Score=%d", approved, score)

             # Decision Logic
            if approved:
                logging.info("Report approved. Finalizing report.")

                return {
                    "review_feedback": feedback,
                    "final_report": report,
                    "revision_count": revision_count
                }
            
            if revision_count >= self.MAX_REVISIONS:
                logging.warning("Max revisions reached. Accepting current report.")

                return {
                    "review_feedback": feedback,
                    "final_report": report,
                    "revision_count": revision_count
                }

            # Needs revision
            logging.info("Report requires revision. Routing back to WriterAgent.")

            return {
                "review_feedback": feedback,
                "revision_count": revision_count + 1
            }

        except Exception as e:
            logging.exception("Error during ReviewerAgent execution.")
            raise CustomException(e, sys)