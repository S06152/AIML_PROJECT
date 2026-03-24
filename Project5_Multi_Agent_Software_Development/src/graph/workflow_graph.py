import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Optional
from langgraph.graph import StateGraph, START, END
from src.models.state import DevTeamState
from src.agents.product_manager import ProductManagerAgent
from src.agents.architect import ArchitectAgent
from src.agents.developer import DeveloperAgent
from src.agents.qa import QAAgent
from src.agents.code_reviewer import CodeReviewAgent
from src.llm.llm_provider import LLMProvider

# ---------------------------------------------------------------------------
# LangGraph Node Name Constants (avoids magic strings throughout the file)
# ---------------------------------------------------------------------------
_NODE_PM: str = "ProductManagerAgent"
_NODE_ARCH: str = "ArchitectAgent"
_NODE_DEV: str = "DeveloperAgent"
_NODE_QA: str = "QAAgent"
_NODE_REVIEW: str = "CodeReviewAgent"

class DevTeamWorkflow:
    """
    Orchestrates the AUTOSAR SWS Multi-Agent software development pipeline.

    Responsibilities:
        - Initialize one shared LLM instance via LLMProvider.
        - Instantiate all 5 agent classes with the shared LLM.
        - Build a LangGraph StateGraph connecting agents in sequence.
        - Compile and return the executable graph.
        - Provide an execute() convenience method for the Streamlit UI.

    Usage:
        workflow = DevTeamWorkflow(user_controls_input)
        compiled = workflow.build_graph()
        result = compiled.invoke({"user_request": "...", "autosar_context": "..."})
    """

    def __init__(self, user_contols_input: dict) -> None:
        """
        Initialize the workflow: LLM + all 5 agents.

        Args:
            user_controls_input (dict): UI-sourced configuration dict containing:
                - "GROQ_API_KEY"   : Groq API key.
                - "LLM_Model_Name" : Groq model identifier.
                - "TEMPERATURE"    : LLM temperature.
                - "TOKEN"          : Max output tokens.

        Raises:
            Raises Exception: If LLM or any agent initialization fails.
        """

        try:
            logging.info("Initializing DevTeamWorkflow...")

            # Shared LLM (one instance for all agents)
            logging.info("Initializing LLM Provider...")
            llm_provider = LLMProvider(user_contols_input)
            llm = llm_provider.get_llm()
            logging.info("LLM instance obtained successfully.")

            # Agent Initialization
            logging.info("Initializing ProductManagerAgent...")
            self._pm: ProductManagerAgent = ProductManagerAgent(llm)

            logging.info("Initializing ArchitectAgent...")
            self._arch: ArchitectAgent = ArchitectAgent(llm)

            logging.info("Initializing DeveloperAgent...")
            self._dev: DeveloperAgent = DeveloperAgent(llm)

            logging.info("Initializing QAAgent...")
            self._qa: QAAgent = QAAgent(llm)

            logging.info("Initializing CodeReviewAgent...")
            self._review: CodeReviewAgent = CodeReviewAgent(llm)

            # Compiled graph cache
            self._compiled_graph: Optional[object] = None

            logging.info("All agents initialized successfully.")

        except Exception as e:
            logging.error("Failed to initialize DevTeamWorkflow agents.")
            raise CustomException(e, sys)

    def build_graph(self):
        """
        Build and compile the LangGraph StateGraph (lazy singleton).

        Graph topology:
            START → PM → Architect → Developer → QA → CodeReview → END

        Each node is an agent's execute() method, which reads from
        DevTeamState and returns a partial state dict to merge.

        Returns:
            CompiledStateGraph: Ready-to-invoke compiled LangGraph graph.

        Raises:
            Raises Exception: If graph construction or compilation fails.
        """

        try:
            if self._compiled_graph is not None:
                logging.info("Returning cached compiled DevTeam graph.")
                return self._compiled_graph
            
            logging.info("Building DevTeam workflow graph...")

            # Initialize StateGraph
            logging.info("Creating StateGraph with DevTeamState...")
            graph = StateGraph(DevTeamState)

            # Register agent nodes
            logging.info("Adding ProductManagerAgent node...")
            graph.add_node(_NODE_PM, self._pm.execute)

            logging.info("Adding ArchitectAgent node...")
            graph.add_node(_NODE_ARCH, self._arch.execute)

            logging.info("Adding DeveloperAgent node...")
            graph.add_node(_NODE_DEV, self._dev.execute)

            logging.info("Adding QAAgent node...")
            graph.add_node(_NODE_QA, self._qa.execute)

            logging.info("Adding CodeReviewAgent node...")
            graph.add_node(_NODE_REVIEW, self._review.execute)

            logging.info("All agent nodes added successfully.")

            # Define workflow edges
            logging.info("Defining workflow execution order...")

            graph.add_edge(START, _NODE_PM)
            graph.add_edge(_NODE_PM, _NODE_ARCH)
            graph.add_edge(_NODE_ARCH, _NODE_DEV)
            graph.add_edge(_NODE_DEV, _NODE_QA)
            graph.add_edge(_NODE_QA, _NODE_REVIEW)
            graph.add_edge(_NODE_REVIEW, END)

            logging.info("Workflow edges defined successfully.")

            # Compile graph
            logging.info("Compiling DevTeam workflow graph...")
            self._compiled_graph = graph.compile()

            logging.info("DevTeam workflow compiled successfully.")

            return self._compiled_graph

        except Exception as e:
            logging.error("Error occurred while building DevTeam workflow graph.")
            raise CustomException(e, sys)
    
    def execute(self, user_request: str, autosar_context: str = "") -> DevTeamState:
        """
        Run the complete 5-agent pipeline and return the final state.

        Args:
            user_request    (str): User's software development request.
            autosar_context (str): Retrieved AUTOSAR SWS context from RAG pipeline.
                                   Empty string if no document was uploaded.

        Returns:
            DevTeamState: Final state dict containing all agent outputs:
                          product_spec, architecture, code, tests, review.

        Raises:
            AutosarMASException: If graph execution fails at any agent node.
        """
        try:
            logging.info(
                "DevTeamWorkflow.execute() called. "
                "Request length: %d chars. Context length: %d chars.",
                len(user_request), len(autosar_context),
            )

            graph = self.build_graph()

            initial_state: DevTeamState = {
                "user_request": user_request,
                "autosar_context": autosar_context,
            }

            logging.info("Invoking compiled DevTeam graph.")
            final_state: DevTeamState = graph.invoke(initial_state)

            logging.info("DevTeamWorkflow.execute() completed successfully.")
            return final_state

        except Exception as e:
            logging.error("Error occurred while executing DevTeam workflow graph.")
            raise CustomException(e, sys)
