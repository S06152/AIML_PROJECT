import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Any
from langchain_groq import ChatGroq
from src.agents.base_agent import BaseAgent
from src.models.state import DevTeamState
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# AUTOSAR QA System Prompt
# ---------------------------------------------------------------------------
_QA_SYSTEM_PROMPT: str = """
You are a Senior AUTOSAR QA Engineer specializing in BSW module verification
and validation for safety-critical automotive software (ISO 26262 ASIL-B/D).

Your task is to generate a comprehensive test suite for the provided AUTOSAR
source code.

Your test suite MUST include:
1. **Unit Tests** (pytest format for Python, or CUnit for C):
   - Happy-path tests for every public API function.
   - Error-path tests for all DET error conditions.
   - Boundary condition tests (NULL pointers, max/min values).

2. **Integration Tests**:
   - Tests verifying correct interaction between module components.
   - Mock/stub stubs for lower-layer dependencies (e.g., MemIf, Fee).

3. **AUTOSAR Compliance Tests**:
   - Verify correct Std_ReturnType (E_OK / E_NOT_OK) handling.
   - Verify DET reporting behavior.
   - Verify state machine transitions as per SWS.

4. **Test Report Template**:
   - Test ID, SWS Requirement Reference, Pass/Fail criteria.

Format: Use pytest syntax with clear docstrings per test.
Include fixture definitions and mock declarations at the top.
"""

class QAAgent(BaseAgent):
    """
    Generates AUTOSAR-compliant test suites from DeveloperAgent's source code.

    Input  (from DevTeamState):
        - ``code``: Source code from DeveloperAgent.

    Output (to DevTeamState):
        - ``tests``: Complete test suite with unit + integration + compliance tests.
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize QAAgent with AUTOSAR QA engineer system prompt.

        Args:
            llm (ChatGroq): Shared Groq LLM instance from LLMProvider.
        """

        logging.info("Initializing QAAgent")

        # Initialize the BaseAgent with the QA system prompt
        super().__init__(llm, _QA_SYSTEM_PROMPT)

        logging.info("QAAgent initialized successfully")

    def execute(self, state: DevTeamState) -> dict[str, Any]:
        """
        Generate AUTOSAR test suite from source code.

        Args:
            state (DevTeamState): Shared workflow state.
                Required keys:
                    - ``code``: Source code from Developer agent.

        Returns:
            dict: {"tests": <generated_test_suite_string>}

        Raises:
            AutosarMASException: On LLM invocation failure.
        """

        try:
            logging.info("QAAgent execution started")

            # Generate tests using the LLM
            tests = self.run(state["code"])

            logging.info("Test generation completed successfully")

            # Return tests in expected workflow state format
            return {"tests": tests}

        except Exception as e:
            logging.error("Error occurred during test generation in QAAgent")
            raise CustomException(e, sys)