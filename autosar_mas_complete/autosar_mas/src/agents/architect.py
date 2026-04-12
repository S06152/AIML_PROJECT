"""
architect.py — ArchitectAgent for AUTOSAR system architecture design.

Receives the product specification from ProductManagerAgent and produces
a comprehensive AUTOSAR-compliant system architecture document.
"""

import sys
from typing import Any
from langchain_groq import ChatGroq
from src.agents.base_agent import BaseAgent
from src.models.state import DevTeamState
from src.utils.logger import logger
from src.utils.exception import CustomException

# ---------------------------------------------------------------------------
# AUTOSAR Architect System Prompt
# ---------------------------------------------------------------------------
_ARCH_SYSTEM_PROMPT: str = """
You are a Senior AUTOSAR Software Architect with 15 years of experience in
BSW (Basic Software) module design for automotive ECUs.

Your task is to produce a comprehensive AUTOSAR-compliant System Architecture
Document from the given product specification.

Your output MUST include:
1. **Module Layered Architecture**: ASCII diagram showing AUTOSAR layers
   (Application / RTE / BSW / MCAL) and module placement.
2. **Component Decomposition**: Sub-modules with responsibilities and interfaces.
3. **API Design**: Function prototypes using AUTOSAR naming conventions
   (e.g., Std_ReturnType Com_SendSignal(Com_SignalIdType SignalId, ...)).
4. **Data Flow**: How data moves between components and memory sections.
5. **Technology Stack**: C standard (MISRA-C:2012), compiler flags, AUTOSAR version.
6. **Memory Layout**: RAM / ROM / NvM block organization.
7. **Error and Diagnostic Strategy**: DET module usage, DEM event mapping.

Format: Markdown with ASCII diagrams where helpful.
Reference AUTOSAR SWS clause numbers where applicable.
"""


class ArchitectAgent(BaseAgent):
    """
    Designs an AUTOSAR-compliant system architecture from a product specification.

    Input  (from DevTeamState):
        - ``product_spec``: AUTOSAR product specification from ProductManagerAgent.

    Output (to DevTeamState):
        - ``architecture``: Detailed AUTOSAR architecture document.
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize ArchitectAgent with AUTOSAR architect system prompt.

        Args:
            llm (ChatGroq): Shared Groq LLM instance from LLMProvider.
        """
        logger.info("Initializing ArchitectAgent.")
        super().__init__(llm, _ARCH_SYSTEM_PROMPT)
        logger.info("ArchitectAgent ready.")

    def execute(self, state: DevTeamState) -> dict[str, Any]:
        """
        Generate the AUTOSAR system architecture from the product specification.

        Args:
            state (DevTeamState): Shared workflow state.
                Required keys:
                    - ``product_spec``: Product specification from PM agent.

        Returns:
            dict: {"architecture": <architecture_document_string>}

        Raises:
            CustomException: On LLM invocation failure.
        """
        try:
            logger.info("ArchitectAgent.execute() started.")

            product_spec: str = state.get("product_spec", "")
            architecture: str = self.run(product_spec)

            logger.info("Architecture generated. Length: %d chars.", len(architecture))
            return {"architecture": architecture}

        except Exception as e:
            raise CustomException(e, sys) from e
