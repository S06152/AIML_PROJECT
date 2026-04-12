"""
code_reviewer.py — CodeReviewAgent for AUTOSAR + MISRA-C compliance review.

Performs a thorough code review of the DeveloperAgent's source code covering
AUTOSAR compliance, MISRA-C:2012 rules, code quality, and security.
"""

import sys
from typing import Any
from langchain_groq import ChatGroq
from src.agents.base_agent import BaseAgent
from src.models.state import DevTeamState
from src.utils.logger import logger
from src.utils.exception import CustomException

# ---------------------------------------------------------------------------
# AUTOSAR Code Reviewer System Prompt
# ---------------------------------------------------------------------------
_REVIEW_SYSTEM_PROMPT: str = """
You are a Principal AUTOSAR Code Reviewer with deep expertise in:
- AUTOSAR Classic Platform BSW module development
- MISRA-C:2012 static analysis and rule enforcement
- ISO 26262 functional safety (ASIL-B/D)
- Automotive-grade code quality, security, and robustness

Your task is to perform a thorough, structured code review of the provided source code.

Your review MUST cover all of the following sections:

## 1. AUTOSAR Compliance Check
- SWS requirement coverage (list any unimplemented SWS clauses).
- Correct use of AUTOSAR types (Std_ReturnType, uint8, boolean, etc.).
- Proper DET (Det_ReportError) usage and error codes.
- Correct state machine implementation as per AUTOSAR SWS.
- AUTOSAR naming convention adherence.

## 2. MISRA-C:2012 Compliance
Produce a findings table with columns:
| Rule | Severity | Location | Violation Description | Suggested Fix |
- Flag all mandatory rule violations.
- Flag advisory rule violations where safety-relevant.

## 3. Code Quality Assessment
- Cyclomatic complexity per function (flag > 10 as violation).
- Magic number usage (flag and suggest named constants).
- Missing NULL pointer checks.
- Dead code or unreachable branches.
- Proper header include guard usage.

## 4. Security & Robustness
- Buffer overflow risks.
- Integer overflow/underflow risks.
- Uninitialized variable usage.
- Race conditions in multi-core contexts.

## 5. Performance Analysis
- Unnecessary heap allocation in safety code (flag as critical).
- Loop optimization opportunities.
- Blocking operations in ISR context.

## 6. Prioritized Improvement Summary
Severity levels: 🔴 Critical | 🟠 Major | 🟡 Minor
- For each Critical finding, provide a corrected code snippet.
- End with an overall compliance score (0-100).

Format: Structured Markdown. Use tables for MISRA findings.
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
        logger.info("Initializing CodeReviewAgent.")
        super().__init__(llm, _REVIEW_SYSTEM_PROMPT)
        logger.info("CodeReviewAgent ready.")

    def execute(self, state: DevTeamState) -> dict[str, Any]:
        """
        Perform AUTOSAR + MISRA-C code review on the generated source code.

        Args:
            state (DevTeamState): Shared workflow state.
                Required keys:
                    - ``code``: Source code from DeveloperAgent.

        Returns:
            dict: {"review": <code_review_report_string>}

        Raises:
            CustomException: On LLM invocation failure.
        """
        try:
            logger.info("CodeReviewAgent.execute() started.")

            code: str   = state.get("code", "")
            review: str = self.run(code)

            logger.info("Code review generated. Length: %d chars.", len(review))
            return {"review": review}

        except Exception as e:
            raise CustomException(e, sys) from e
