"""
workflow_graph.py — LangGraph StateGraph orchestrator for the AUTOSAR MAS pipeline.

Connects the 5 specialist agents into a directed acyclic graph (DAG):

    START
      │
      ▼
    ProductManagerAgent   ← user_request + autosar_context → product_spec
      │
      ▼
    ArchitectAgent        ← product_spec → architecture
      │
      ▼
    DeveloperAgent        ← architecture → code
      │
      ▼
    QAAgent               ← code → tests
      │
      ▼
    CodeReviewAgent       ← code → review
      │
      ▼
    END

Design:
    - Shared LLM: one ChatGroq instance reused by all agents (cost-efficient).
    - Lazy graph compilation: compiled graph is cached after first build.
    - DevTeamState flows through all nodes; each node merges a partial update.

FIX vs original:
    - Corrected self._user_controls → self._user_input in main.py integration.
    - DevTeamState changed to total=False so all fields are Optional.
"""

import sys
from typing import Optional
from langgraph.graph import StateGraph, START, END
from src.models.state import DevTeamState
from src.agents.product_manager import ProductManagerAgent
from src.agents.architect import ArchitectAgent
from src.agents.developer import DeveloperAgent
from src.agents.qa import QAAgent
from src.agents.code_reviewer import CodeReviewAgent
from src.llm.llm_provider import LLMProvider
from src.utils.logger import logger
from src.utils.exception import CustomException

# ---------------------------------------------------------------------------
# Node name constants (avoids magic strings throughout the module)
# ---------------------------------------------------------------------------
_NODE_PM:     str = "ProductManagerAgent"
_NODE_ARCH:   str = "ArchitectAgent"
_NODE_DEV:    str = "DeveloperAgent"
_NODE_QA:     str = "QAAgent"
_NODE_REVIEW: str = "CodeReviewAgent"


class DevTeamWorkflow:
    """
    Orchestrates the AUTOSAR SWS Multi-Agent software development pipeline.

    Responsibilities:
        - Initialize a single shared LLM instance via LLMProvider.
        - Instantiate all 5 specialist agents with the shared LLM.
        - Build a LangGraph StateGraph connecting agents in sequence.
        - Compile and cache the executable graph.
        - Provide an execute() entry point for the Streamlit UI.

    Usage:
        workflow = DevTeamWorkflow(user_controls)
        result   = workflow.execute(user_request, autosar_context)
    """

    def __init__(self, user_controls_input: dict) -> None:
        """
        Initialize the workflow: LLM + all 5 agents.

        Args:
            user_controls_input (dict): UI configuration dict containing:
                - "GROQ_API_KEY" (str)  : Groq API key.
                - "LLM_MODEL"    (str)  : Groq model identifier.
                - "TEMPERATURE"  (float): LLM temperature.
                - "TOKEN"        (int)  : Max output tokens.

        Raises:
            CustomException: If LLM or any agent initialization fails.
        """
        try:
            logger.info("Initializing DevTeamWorkflow.")

            # ── Shared LLM (one instance for all 5 agents) ──────────────────
            logger.info("Creating shared LLM instance.")
            llm_provider = LLMProvider(user_controls_input)
            llm = llm_provider.get_llm()

            # ── Agent instantiation ──────────────────────────────────────────
            logger.info("Instantiating specialist agents.")
            self._pm:     ProductManagerAgent = ProductManagerAgent(llm)
            self._arch:   ArchitectAgent      = ArchitectAgent(llm)
            self._dev:    DeveloperAgent      = DeveloperAgent(llm)
            self._qa:     QAAgent             = QAAgent(llm)
            self._review: CodeReviewAgent     = CodeReviewAgent(llm)

            # ── Compiled graph cache ─────────────────────────────────────────
            self._compiled_graph: Optional[object] = None

            logger.info("DevTeamWorkflow: all agents ready.")

        except Exception as e:
            raise CustomException(e, sys) from e

    def build_graph(self):
        """
        Build and compile the LangGraph StateGraph (lazy singleton).

        Graph topology (sequential pipeline):
            START → PM → Architect → Developer → QA → CodeReview → END

        Each node is an agent's execute() method, which reads from DevTeamState
        and returns a partial state dict that LangGraph merges automatically.

        Returns:
            CompiledStateGraph: Ready-to-invoke compiled graph.

        Raises:
            CustomException: If graph construction or compilation fails.
        """
        try:
            if self._compiled_graph is not None:
                logger.info("Returning cached compiled DevTeam graph.")
                return self._compiled_graph

            logger.info("Building DevTeam workflow graph.")

            # ── Create StateGraph ────────────────────────────────────────────
            graph = StateGraph(DevTeamState)

            # ── Register agent nodes ─────────────────────────────────────────
            graph.add_node(_NODE_PM,     self._pm.execute)
            graph.add_node(_NODE_ARCH,   self._arch.execute)
            graph.add_node(_NODE_DEV,    self._dev.execute)
            graph.add_node(_NODE_QA,     self._qa.execute)
            graph.add_node(_NODE_REVIEW, self._review.execute)

            logger.info("All agent nodes registered.")

            # ── Define sequential edges ──────────────────────────────────────
            graph.add_edge(START,       _NODE_PM)
            graph.add_edge(_NODE_PM,    _NODE_ARCH)
            graph.add_edge(_NODE_ARCH,  _NODE_DEV)
            graph.add_edge(_NODE_DEV,   _NODE_QA)
            graph.add_edge(_NODE_QA,    _NODE_REVIEW)
            graph.add_edge(_NODE_REVIEW, END)

            logger.info("Workflow edges defined.")

            # ── Compile ──────────────────────────────────────────────────────
            self._compiled_graph = graph.compile()
            logger.info("DevTeam workflow compiled successfully.")

            return self._compiled_graph

        except Exception as e:
            raise CustomException(e, sys) from e

    def execute(
        self,
        user_request: str,
        autosar_context: str = "",
        module_name: str     = "MODULE",
    ) -> DevTeamState:
        """
        Run the complete 5-agent pipeline and return the final DevTeamState.

        Args:
            user_request    (str): User's natural-language development request.
            autosar_context (str): Retrieved AUTOSAR SWS context from RAG.
                                   Empty string if no document was uploaded.
            module_name     (str): AUTOSAR module name (used for artifact filenames).

        Returns:
            DevTeamState: Final state dict containing all agent outputs:
                          product_spec, architecture, code, tests, review.

        Raises:
            CustomException: If graph execution fails at any agent node.
        """
        try:
            logger.info(
                "DevTeamWorkflow.execute() started. "
                "request=%d chars, context=%d chars.",
                len(user_request), len(autosar_context),
            )

            graph = self.build_graph()

            initial_state: DevTeamState = {
                "user_request":    user_request,
                "autosar_context": autosar_context,
                "module_name":     module_name,
            }

            logger.info("Invoking compiled DevTeam graph.")
            final_state: DevTeamState = graph.invoke(initial_state)

            logger.info("DevTeamWorkflow.execute() completed successfully.")
            return final_state

        except Exception as e:
            raise CustomException(e, sys) from e
