"""
developer.py — DeveloperAgent for AUTOSAR source code generation.

Receives the system architecture from ArchitectAgent and generates
production-ready, MISRA-C:2012 compliant C source code.
"""

import sys
from typing import Any
from langchain_groq import ChatGroq
from src.agents.base_agent import BaseAgent
from src.models.state import DevTeamState
from src.utils.logger import logger
from src.utils.exception import CustomException

# ---------------------------------------------------------------------------
# AUTOSAR Developer System Prompt
# ---------------------------------------------------------------------------
_DEV_SYSTEM_PROMPT: str = """
You are a Senior AUTOSAR C Developer specializing in BSW module implementation
for automotive-grade embedded systems (ISO 26262 ASIL-B/D).

Your task is to generate production-ready, AUTOSAR-compliant source code
from the provided system architecture document.

Your code MUST:
1. Follow MISRA-C:2012 guidelines (annotate any necessary deviations).
2. Use AUTOSAR standard types: uint8, uint16, uint32, Std_ReturnType, boolean, etc.
3. Implement proper DET (Default Error Tracer) checks at every function entry.
4. Include full Doxygen-style header comments for every function.
5. Be organized into:
   a) A header file (.h) with include guards and all API declarations.
   b) A source file (.c) with full implementation, DET checks, state logic.
6. Use AUTOSAR naming conventions: <Module>_<Action><Object>() (e.g., Com_SendSignal).
7. Handle ALL SWS-specified error conditions with E_OK / E_NOT_OK return codes.
8. Use #define named constants — no magic numbers.

Output format:
- First: The complete .h header file in a ```c ... ``` code block.
- Then: The complete .c source file in a ```c ... ``` code block.
- Include a brief explanation of key design decisions after the code.
"""


class DeveloperAgent(BaseAgent):
    """
    Generates AUTOSAR-compliant C source code from the system architecture.

    Input  (from DevTeamState):
        - ``architecture``: System architecture from ArchitectAgent.

    Output (to DevTeamState):
        - ``code``: Complete C source code (header + implementation).
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize DeveloperAgent with AUTOSAR developer system prompt.

        Args:
            llm (ChatGroq): Shared Groq LLM instance from LLMProvider.
        """
        logger.info("Initializing DeveloperAgent.")
        super().__init__(llm, _DEV_SYSTEM_PROMPT)
        logger.info("DeveloperAgent ready.")

    def execute(self, state: DevTeamState) -> dict[str, Any]:
        """
        Generate AUTOSAR-compliant source code from the system architecture.

        Args:
            state (DevTeamState): Shared workflow state.
                Required keys:
                    - ``architecture``: Architecture document from ArchitectAgent.

        Returns:
            dict: {"code": <source_code_string>}

        Raises:
            CustomException: On LLM invocation failure.
        """
        try:
            logger.info("DeveloperAgent.execute() started.")

            architecture: str = state.get("architecture", "")
            code: str = self.run(architecture)

            logger.info("Source code generated. Length: %d chars.", len(code))
            return {"code": code}

        except Exception as e:
            raise CustomException(e, sys) from e
