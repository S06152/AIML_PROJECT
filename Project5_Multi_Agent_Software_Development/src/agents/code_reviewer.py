import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Any
from langchain_groq import ChatGroq
from src.agents.base_agent import BaseAgent
from src.models.state import DevTeamState

# ---------------------------------------------------------------------------
# AUTOSAR Code Reviewer System Prompt
# ---------------------------------------------------------------------------
_REVIEW_SYSTEM_PROMPT: str = """
You are a Principal AUTOSAR Code Reviewer with deep expertise in:
- AUTOSAR Classic Platform BSW module development
- MISRA-C:2012 static analysis
- ISO 26262 functional safety (ASIL-B/D)
- Automotive-grade code quality and security

Your task is to perform a thorough code review of the provided AUTOSAR source code.

Your review MUST cover:
1. **AUTOSAR Compliance**:
   - SWS requirement coverage (list any unimplemented SWS clauses).
   - Correct use of AUTOSAR types (Std_ReturnType, uint8, etc.).
   - Proper DET (Det_ReportError) usage.
   - Correct state machine implementation per SWS.

2. **MISRA-C:2012 Compliance**:
   - List any MISRA rule violations found.
   - Suggest compliant alternatives.

3. **Code Quality**:
   - Naming convention adherence (AUTOSAR standard).
   - Function complexity (max cyclomatic complexity = 10 for safety code).
   - Magic numbers / hardcoded values — flag and suggest named constants.
   - Missing NULL pointer checks.

4. **Security & Robustness**:
   - Buffer overflows, integer overflows, unreachable code.
   - Uninitialized variables.

5. **Performance**:
   - Unnecessary memory allocation.
   - Loop optimization opportunities.

6. **Improvement Suggestions**:
   - Prioritized list: Critical → Major → Minor.
   - Provide corrected code snippets for critical issues.

Format: Structured Markdown with severity ratings per finding.
Use tables for MISRA findings: Rule | Severity | Location | Suggestion.
"""

class CodeReviewAgent(BaseAgent):
    """
    Performs AUTOSAR + MISRA-C compliance code review on generated source code.

    Input  (from DevTeamState):
        - ``code``: Source code from DeveloperAgent.

    Output (to DevTeamState):
        - ``review``: Structured code review report with severity-tagged findings.
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize CodeReviewAgent with AUTOSAR reviewer system prompt.

        Args:
            llm (ChatGroq): Shared Groq LLM instance from LLMProvider.
        """

        logging.info("Initializing CodeReviewAgent")

        # Initialize the BaseAgent with system prompt
        super().__init__(llm, _REVIEW_SYSTEM_PROMPT)

        logging.info("CodeReviewAgent initialized successfully")

    def execute(self, state: DevTeamState) -> dict[str, Any]:
        """
        Perform AUTOSAR + MISRA-C code review on generated source code.

        Args:
            state (DevTeamState): Shared workflow state.
                Required keys:
                    - ``code``: Source code from Developer agent.

        Returns:
            dict: {"review": <structured_review_report_string>}

        Raises:
            AutosarMASException: On LLM invocation failure.
        """

        try:
            logging.info("CodeReviewAgent execution started")

            # Run the LLM review process
            review = self.run(state["code"])

            logging.info("Code review generated successfully")

            # Return result in expected state format
            return {"review": review}

        except Exception as e:
            logging.error("Error occurred during code review process")
            raise CustomException(e, sys)