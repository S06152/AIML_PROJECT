import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Optional, Dict, Any
from src.llm.llm_provider import LLMProvider
from src.models.state import ResearchState
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.search_agent import SearchAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.writer_agent import WriterAgent
from src.agents.reviewer_agent import ReviewerAgent
from langgraph.graph import StateGraph, START, END
import warnings
warnings.filterwarnings("ignore")

# Routing Logic After Review Agent
def route_after_review(state: Dict[str, Any]) -> str:
    """
    Decide next step after review agent.

    Returns:
        "rewrite" → send back to writer
        "end"     → finish workflow
    """
    try:
        logging.info("Evaluating review feedback for routing...")

        feedback = state.get("review_feedback")

        if not feedback:
            logging.warning("No review feedback found → Ending workflow.")
            return "end"

        if not feedback.get("approved", False):
            logging.info("Review NOT approved → Routing back to writer.")
            return "rewrite"

        logging.info("Review approved → Ending workflow.")
        return "end"

    except Exception:
        logging.exception("Error in routing logic → Defaulting to END.")
        return "end"

class GraphBuilder:
    """
    Builds and executes the Multi-Agent Research & Report Generation workflow.

    Agents:
        1. Orchestrator Agent → Generates plan & queries
        2. Search Agent       → Fetches web results
        3. Extraction Agent   → Extracts structured insights
        4. Writer Agent       → Writes report
        5. Reviewer Agent     → Reviews & improves report
    """

    def __init__(self, user_contols_input: dict) -> None:
        """
        Initialize LLM and all agents.

        Args:
            user_controls_input (dict): UI configuration

        Raises:
            CustomException: If initialization fails
        """

        try:
            logging.info("GRAPH BUILDER INITIALIZATION START")

            # Initialize shared LLM
            logging.info("Initializing LLM Provider...")
            llm_provider = LLMProvider(user_contols_input)
            llm = llm_provider.get_llm()
            logging.info("LLM instance initialized  successfully.")

            # Initialize Agents
            logging.info("Initializing Orchestrator Agent...")
            self._orchestrator = OrchestratorAgent(llm)

            logging.info("Initializing Search Agent...")
            self._searcher = SearchAgent(llm)

            logging.info("Initializing Extraction Agent...")
            self._extractor = ExtractionAgent(llm)

            logging.info("Initializing Writer Agent...")
            self._writer = WriterAgent(llm)

            logging.info("Initializing Reviewer Agent...")
            self._reviewer = ReviewerAgent(llm)

            # Compiled graph cache
            self._compiled_graph: Optional[Any] = None

            logging.info("All agents initialized successfully.")

        except Exception as e:
            logging.exception("ERROR during GraphBuilder initialization.")
            raise CustomException(e, sys)

    def build_graph(self):
        """
        Build and compile LangGraph workflow.

        Flow:
            START → orchestrate → search → extract → write → review → (loop or END)

        Returns:
            Compiled graph
        """

        try:
            # Return cached graph
            if self._compiled_graph is not None:
                logging.info("Returning cached compiled workflow  graph.")
                return self._compiled_graph
            
            logging.info("Building workflow graph...")

            # Create Graph
            graph = StateGraph(ResearchState)

           # Add Nodes (Agents)
            graph.add_node("orchestrate", self._orchestrator.execute)
            graph.add_node("search", self._searcher.execute)
            graph.add_node("extract", self._extractor.execute)
            graph.add_node("write", self._writer.execute)
            graph.add_node("review", self._reviewer.execute)

            logging.info("All agent nodes added successfully.")

            # Define Edges
            graph.add_edge(START, "orchestrate")
            graph.add_edge("orchestrate", "search")
            graph.add_edge("search", "extract")
            graph.add_edge("extract", "write")
            graph.add_edge("write", "review")

            # Conditional loop
            graph.add_conditional_edges(
                "review",
                route_after_review,
                {
                    "rewrite": "write",
                    "end": END,
                }
            )

            logging.info("Workflow edges defined successfully.")

            # Compile Graph
            self._compiled_graph = graph.compile()
            logging.info("Workflow graph compiled successfully.")

            return self._compiled_graph

        except Exception as e:
            logging.exception("ERROR while building workflow graph.")
            raise CustomException(e, sys)
    
    def execute(self, topic: str):
        """
        Execute the full multi-agent pipeline.

        Args:
            topic (str): User input topic

        Returns:
            Dict[str, Any]: Final workflow state
        """

        try:
            logging.info("WORKFLOW EXECUTION START")
            logging.info(f"User topic: {topic}")

            graph = self.build_graph()

            # Initial state
            initial_state: Dict[str, Any] = {
                "topic": topic,
                "research_plan": "",
                "search_queries": [],
                "raw_search_results": [],
                "extracted_data": None,
                "draft_report": "",
                "final_report": "",
                "review_feedback": None,
                "revision_count": 0,
                "errors": [],
                "messages": [],
            }

            logging.info("Invoking workflow graph...")
            final_state = graph.invoke(initial_state)

            logging.info("Workflow executed successfully.")
            logging.info("WORKFLOW EXECUTION END")

            return final_state

        except Exception as e:
            logging.exception("ERROR during workflow execution.")
            raise CustomException(e, sys)