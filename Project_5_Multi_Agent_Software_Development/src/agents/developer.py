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
# AUTOSAR Developer System Prompt
# ---------------------------------------------------------------------------
_DEV_SYSTEM_PROMPT: str = """
You are a Senior AUTOSAR C/Python Developer specializing in BSW module
implementation for automotive-grade embedded systems.

Your task is to generate production-ready, AUTOSAR-compliant source code
based on the provided system architecture document.

Your code MUST:
1. Follow MISRA-C:2012 guidelines (annotate any necessary deviations).
2. Use AUTOSAR standard types: uint8, uint16, uint32, Std_ReturnType, etc.
3. Implement proper DET (Default Error Tracer) checks at function entry.
4. Include full Doxygen-style header comments for every function.
5. Organize into: header file (.h) with declarations + source file (.c) with implementation.
6. Include #include guards in header files.
7. Use AUTOSAR naming conventions: <Module>_<Action><Object>().
8. Handle all SWS-specified error conditions.

Output format:
- First: The .h header file (complete, with include guards and API declarations).
- Then: The .c source file (complete implementation with DET checks and logic).
- Use markdown code blocks with language tags (```c ... ```).
"""

class DeveloperAgent(BaseAgent):
    """
    Generates AUTOSAR-compliant source code from the system architecture.

    Input  (from DevTeamState):
        - ``architecture``: System architecture from ArchitectAgent.

    Output (to DevTeamState):
        - ``code``: Complete C/Python source code (header + implementation).
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize DeveloperAgent with AUTOSAR developer system prompt.

        Args:
            llm (ChatGroq): Shared Groq LLM instance from LLMProvider.
        """

        logging.info("Initializing DeveloperAgent")

        # Initialize the BaseAgent with the system prompt
        super().__init__(llm, _DEV_SYSTEM_PROMPT)

        logging.info("DeveloperAgent initialized successfully")

    def execute(self, state: DevTeamState) -> dict[str, Any]:
        """
        Generate AUTOSAR-compliant source code from system architecture.

        Args:
            state (DevTeamState): Shared workflow state.
                Required keys:
                    - ``architecture``: Architecture document from Architect agent.

        Returns:
            dict: {"code": <generated_source_code_string>}

        Raises:
            AutosarMASException: On LLM invocation failure.
        """

        try:
            logging.info("DeveloperAgent execution started")

            # Generate backend code using the LLM
            code = self.run(state["architecture"])

            logging.info("Code generation completed successfully")

            # Return result in workflow state format
            return {"code": code}

        except Exception as e:
            logging.error("Error occurred during code generation in DeveloperAgent")
            raise CustomException(e, sys)