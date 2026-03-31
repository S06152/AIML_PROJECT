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
# AUTOSAR Architect System Prompt
# ---------------------------------------------------------------------------
_ARCH_SYSTEM_PROMPT: str = """
You are a Senior AUTOSAR Software Architect with 15 years of experience in
BSW (Basic Software) module design for automotive ECUs.

Your task is to produce a comprehensive AUTOSAR-compliant System Architecture
Document from the given product specification.

Your output MUST include:
1. **Module Layered Architecture**: Diagram (text-based) showing AUTOSAR layers
   (Application / RTE / BSW / MCAL) and module placement.
2. **Component Decomposition**: Sub-modules with responsibilities and interfaces.
3. **API Design**: Function prototypes with AUTOSAR naming conventions
   (e.g., Std_ReturnType NvM_ReadBlock(NvM_BlockIdType BlockId, ...)).
4. **Data Flow**: How data moves between components and memory sections.
5. **Technology Stack**: C standard (MISRA-C:2012), compiler, AUTOSAR version.
6. **Memory Layout**: RAM / ROM / NvM block organization.
7. **Error and Diagnostic Strategy**: DET, DEM event mapping.

Format: Markdown with ASCII diagrams where helpful.
Reference AUTOSAR SWS clause numbers where applicable.
"""

class ArchitectAgent(BaseAgent):
    """
    Designs AUTOSAR-compliant system architecture from product specification.

    Input  (from DevTeamState):
        - ``product_spec``: AUTOSAR product specification from ProductManagerAgent.

    Output (to DevTeamState):
        - ``architecture``: Detailed AUTOSAR architecture document.
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize ArchitectAgent with AUTOSAR-aware architect system prompt.

        Args:
            llm (ChatGroq): Shared Groq LLM instance from LLMProvider.
        """

        logging.info("Initializing ArchitectAgent")
        super().__init__(llm, _ARCH_SYSTEM_PROMPT)

        logging.info("ArchitectAgent initialized successfully")

    def execute(self, state: DevTeamState) -> dict[str, Any]:
        """
        Generate system architecture from the AUTOSAR product specification.

        Args:
            state (DevTeamState): Shared workflow state.
                Required keys:
                    - ``product_spec``: Product specification from PM agent.

        Returns:
            dict: {"architecture": <generated_architecture_string>}

        Raises:
            Raises Exception: On LLM invocation failure.
        """

        try:
            logging.info("ArchitectAgent execution started")

            # Run LLM to generate architecture
            architecture = self.run(state["product_spec"])

            logging.info("Architecture generation completed successfully")

            return {"architecture": architecture}

        except Exception as e:
            logging.error("Error occurred in ArchitectAgent execution")
            raise CustomException(e, sys)