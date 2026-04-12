"""
state.py — Shared workflow state model for the AUTOSAR MAS LangGraph pipeline.

DevTeamState is the single source of truth that flows through all agent nodes.
Each agent reads from this state and writes back a partial update dict.

Design:
    - TypedDict: enables static type checking and IDE autocomplete.
    - Optional fields: agents only populate their own output field.
    - LangGraph merges partial dicts returned by each node automatically.
"""

from typing import TypedDict, Optional
from src.utils.logger import logger

logger.info("Loading DevTeamState schema.")


class DevTeamState(TypedDict, total=False):
    """
    Shared state container for the AUTOSAR SWS 5-agent LangGraph workflow.

    Field ownership:
    ┌──────────────────────┬──────────────────────────────────────────────┐
    │  Field               │  Producing Component                         │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │  user_request        │  User input (initial seed)                   │
    │  autosar_context     │  RAG pipeline (QAChain retrieval)            │
    │  module_name         │  Extracted from user_request (heuristic)     │
    │  product_spec        │  ProductManagerAgent                         │
    │  architecture        │  ArchitectAgent                              │
    │  code                │  DeveloperAgent                              │
    │  tests               │  QAAgent                                     │
    │  review              │  CodeReviewAgent                             │
    └──────────────────────┴──────────────────────────────────────────────┘

    Notes:
        - All fields are Optional (total=False) because LangGraph merges
          partial state dicts returned by each node.
        - Agents should never read fields they did not produce unless
          explicitly documented in their docstring.
    """

    # ── Inputs ──────────────────────────────────────────────────────────────
    user_request: str           # User's natural-language development request
    autosar_context: str        # Retrieved AUTOSAR SWS specification context
    module_name: str            # Extracted AUTOSAR module name (e.g., "COM", "NvM")

    # ── Agent Outputs ────────────────────────────────────────────────────────
    product_spec: str           # AUTOSAR-compliant product requirements spec
    architecture: str           # System architecture and API design document
    code: str                   # MISRA-C/Python source code implementation
    tests: str                  # pytest / CUnit test suite
    review: str                 # Code review report with severity ratings
