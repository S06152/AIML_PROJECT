"""
product_manager.py — ProductManagerAgent for AUTOSAR SWS requirements generation.

Converts a user's natural-language request + retrieved AUTOSAR SWS context
into a structured, AUTOSAR-compliant Product Specification Document.

FIX vs original:
    - Original had a critical bug: state[combined_input] (using a string as a key).
    - Fixed to: self.run(combined_input) — passing the string as input to the LLM.
    - Removed redundant double LLM calls (spec = self.run(state["user_request"])
      then context = self.run(state["autosar_context"]) was running separate LLM
      calls, not combining them correctly).
"""

import sys
from typing import Any
from langchain_groq import ChatGroq
from src.agents.base_agent import BaseAgent
from src.models.state import DevTeamState
from src.utils.logger import logger
from src.utils.exception import CustomException

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
   referencing relevant SWS clauses (e.g., [SWS_COM_00001]).
3. **Non-Functional Requirements**: Performance, memory, timing constraints.
4. **API Endpoints / SWS Interface**: Function signatures as per the AUTOSAR SWS.
5. **Error Handling Requirements**: Required Dem events, DET codes.
6. **Compliance Notes**: AUTOSAR version, BSW module, schema references.

Format: Use Markdown with clear section headers.
Tone: Precise, technical, standards-compliant.
Base all requirements strictly on the AUTOSAR SWS context provided.
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
        Initialize ProductManagerAgent with AUTOSAR PM system prompt.

        Args:
            llm (ChatGroq): Shared Groq LLM instance from LLMProvider.
        """
        logger.info("Initializing ProductManagerAgent.")
        super().__init__(llm, _PM_SYSTEM_PROMPT)
        logger.info("ProductManagerAgent ready.")

    def execute(self, state: DevTeamState) -> dict[str, Any]:
        """
        Generate an AUTOSAR-compliant product specification.

        Combines user_request and autosar_context into a single grounded
        prompt, then invokes the LLM once to produce the specification.

        Args:
            state (DevTeamState): Shared workflow state.
                Required keys:
                    - ``user_request``   : User's software requirement.
                    - ``autosar_context``: AUTOSAR SWS context from RAG.

        Returns:
            dict: {"product_spec": <specification_string>}

        Raises:
            CustomException: On LLM invocation failure.
        """
        try:
            logger.info("ProductManagerAgent.execute() started.")

            user_request: str   = state.get("user_request", "")
            autosar_context: str = state.get("autosar_context", "")

            # FIX: Build combined input string, then call self.run() once
            combined_input: str = (
                f"## User Request\n{user_request}\n\n"
                f"## AUTOSAR SWS Context (from uploaded specification)\n"
                f"{autosar_context if autosar_context else 'No AUTOSAR document uploaded.'}"
            )

            product_spec: str = self.run(combined_input)

            logger.info("Product specification generated. Length: %d chars.", len(product_spec))
            return {"product_spec": product_spec}

        except Exception as e:
            raise CustomException(e, sys) from e
