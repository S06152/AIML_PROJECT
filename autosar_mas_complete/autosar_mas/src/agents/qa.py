"""
qa.py — QAAgent for AUTOSAR test case generation.

Receives the source code from DeveloperAgent and generates a comprehensive
test suite covering unit tests, integration tests, and AUTOSAR compliance tests.
"""

import sys
from typing import Any
from langchain_groq import ChatGroq
from src.agents.base_agent import BaseAgent
from src.models.state import DevTeamState
from src.utils.logger import logger
from src.utils.exception import CustomException

# ---------------------------------------------------------------------------
# AUTOSAR QA Engineer System Prompt
# ---------------------------------------------------------------------------
_QA_SYSTEM_PROMPT: str = """
You are a Senior AUTOSAR QA Engineer specializing in BSW module verification
and validation for safety-critical automotive software (ISO 26262 ASIL-B/D).

Your task is to generate a comprehensive test suite for the provided AUTOSAR source code.

Your test suite MUST include all of the following sections:

## 1. Unit Tests (pytest format)
- Happy-path tests for every public API function.
- Error-path tests for all DET error conditions (NULL pointers, invalid IDs).
- Boundary condition tests (max/min values, empty buffers).
- Use pytest fixtures and parametrize decorators where appropriate.

## 2. Integration Tests
- Tests verifying correct interaction between module components.
- Mock/stub stubs for lower-layer dependencies using unittest.mock.
- Tests for correct state machine transitions as per AUTOSAR SWS.

## 3. AUTOSAR Compliance Tests
- Verify Std_ReturnType (E_OK / E_NOT_OK) is returned correctly.
- Verify DET reporting behavior (Det_ReportError is called on errors).
- Verify correct initialization sequence (module must be initialized before use).

## 4. Test Report Template (Markdown table)
Columns: Test_ID | SWS_Requirement | Description | Expected_Result | Pass_Fail_Criteria

Format: Use Python pytest syntax throughout.
Include all fixture definitions and mock declarations at the top.
Add clear docstrings to every test function explaining what it validates.
"""


class QAAgent(BaseAgent):
    """
    Generates an AUTOSAR-compliant test suite from DeveloperAgent's source code.

    Input  (from DevTeamState):
        - ``code``: Source code from DeveloperAgent.

    Output (to DevTeamState):
        - ``tests``: Complete test suite (unit + integration + compliance tests).
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize QAAgent with AUTOSAR QA engineer system prompt.

        Args:
            llm (ChatGroq): Shared Groq LLM instance from LLMProvider.
        """
        logger.info("Initializing QAAgent.")
        super().__init__(llm, _QA_SYSTEM_PROMPT)
        logger.info("QAAgent ready.")

    def execute(self, state: DevTeamState) -> dict[str, Any]:
        """
        Generate an AUTOSAR-compliant test suite from the source code.

        Args:
            state (DevTeamState): Shared workflow state.
                Required keys:
                    - ``code``: Source code from DeveloperAgent.

        Returns:
            dict: {"tests": <test_suite_string>}

        Raises:
            CustomException: On LLM invocation failure.
        """
        try:
            logger.info("QAAgent.execute() started.")

            code: str  = state.get("code", "")
            tests: str = self.run(code)

            logger.info("Test suite generated. Length: %d chars.", len(tests))
            return {"tests": tests}

        except Exception as e:
            raise CustomException(e, sys) from e
