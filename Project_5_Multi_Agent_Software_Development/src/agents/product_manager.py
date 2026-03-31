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
# AUTOSAR-aware Product Manager System Prompt
# ---------------------------------------------------------------------------
_PM_SYSTEM_PROMPT: str = """
You are a Senior AUTOSAR Product Manager at a Tier-1 automotive supplier.

Your task is to convert a user's software request and AUTOSAR SWS specification
context into a detailed, AUTOSAR-compliant Product Specification Document.

Your output MUST include:
1. **Module Overview**: Brief description of the AUTOSAR software module.
2. **Functional Requirements**: Numbered list of functional requirements
   referencing relevant SWS clauses (e.g., [SWS_NvM_00001]).
3. **Non-Functional Requirements**: Performance, memory, timing constraints.
4. **API Endpoints / SWS Interface**: Function signatures as per the AUTOSAR SWS.
5. **Error Handling Requirements**: Required Dem events, DET codes.
6. **Compliance Notes**: AUTOSAR version, BSW module, schema references.

Format: Use Markdown with clear section headers.
Tone: Precise, technical, standards-compliant.
"""

class ProductManagerAgent(BaseAgent):
    """
    Converts user request + AUTOSAR SWS context into a product specification.

    Input  (from DevTeamState):
        - ``user_request``   : User's natural-language feature request.
        - ``autosar_context``: Retrieved AUTOSAR SWS document excerpts (RAG).

    Output (to DevTeamState):
        - ``product_spec``   : Structured AUTOSAR-compliant product specification.
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize ProductManagerAgent with AUTOSAR-aware PM system prompt.

        Args:
            llm (ChatGroq): Shared Groq LLM instance from LLMProvider.
        """

        logging.info("Initializing ProductManagerAgent")

        # Initialize BaseAgent with the system prompt
        super().__init__(llm, _PM_SYSTEM_PROMPT)

        logging.info("ProductManagerAgent initialized successfully")

    def execute(self, state: DevTeamState) -> dict[str, Any]:
        """
        Generate a product specification from user request and AUTOSAR context.

        Args:
            state (DevTeamState): Shared workflow state.
                Required keys:
                    - ``user_request``   : User's software requirement.
                    - ``autosar_context``: AUTOSAR SWS document context (from RAG).

        Returns:
            dict: {"product_spec": <generated_specification_string>}

        Raises:
            AutosarMASException: On LLM invocation failure.
        """

        try:
            logging.info("ProductManagerAgent execution started")

            # Generate product specification using the LLM
            spec = self.run(state["user_request"])
            context = self.run(state["autosar_context"])

            # Combine user request with AUTOSAR context for grounded generation
            combined_input: str = (
                f"## User Request\n{spec}\n\n"
                f"## AUTOSAR SWS Context (from uploaded specification)\n"
                f"{context if context else 'No document uploaded.'}"
            )

            product_spec: str = self.run(state[combined_input])
            logging.info("Product specification generated successfully")

            # Return result in the workflow state format
            return {"product_spec": product_spec}

        except Exception as e:
            logging.error("Error occurred during product specification generation")
            raise CustomException(e, sys)