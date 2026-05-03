import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Any, Dict, List
from langchain_groq import ChatGroq
from src.agents.base_agent import BaseAgent
from src.models.state import ResearchState
import warnings
warnings.filterwarnings("ignore")

# Orchestrator System Prompt
_ORCHESTRATOR_PROMPT: str = """
You are a Research Orchestrator Agent.

Your job is to:
1. Understand the user's topic
2. Break it into a structured research plan
3. Generate multiple focused search queries

Output strictly in this format:

Research Plan:
<clear structured plan>

Search Queries:
- query 1
- query 2
- query 3
- query 4
"""

class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent (Planner)

    Responsibilities:
        - Analyze user topic
        - Generate research plan
        - Generate search queries

    Input:
        state["topic"]

    Output:
        {
            "research_plan": str,
            "search_queries": List[str]
        }
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Initialize OrchestratorAgent.

        Args:
            llm (ChatGroq): Shared LLM instance
        """
        try:
            logging.info("Initializing OrchestratorAgent...")

            super().__init__(llm, _ORCHESTRATOR_PROMPT)

            logging.info("OrchestratorAgent initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize OrchestratorAgent")
            raise CustomException(e, sys)

    def execute(self, state: ResearchState) -> Dict[str, Any]:
        """
        Execute planning step.

        Args:
            state (ResearchState): Shared workflow state

        Returns:
            Dict:
                {
                    "research_plan": str,
                    "search_queries": list[str]
                }
        """

        try:
            logging.info("ORCHESTRATOR EXECUTION START")

            topic: str = state.get("topic", "").strip()
            
            if not topic:
                raise ValueError("Topic is missing or empty in state.")
            
            logging.info("Processing topic: %s", topic)
         
            # Run LLM
            response: str = self.run(topic)

            if not response:
                logging.warning("Empty response from LLM. Using fallback.")
                return {
                    "research_plan": topic,
                    "search_queries": [topic]
                }
        
            logging.info("LLM response received successfully from OrchestratorAgent.")

            # Parse Output
            research_plan: str = ""
            search_queries: List[str] = []

            try:
                if "Search Queries:" in response:
                    parts = response.split("Search Queries:")

                    # Extract plan
                    research_plan = parts[0].replace("Research Plan:", "").strip()

                    # Extract queries
                    queries_block = parts[1].strip().split("\n")

                    search_queries = [q.replace("- ", "").strip() for q in queries_block if q.strip()]

                else:
                    logging.warning("Unexpected LLM response format. Using fallback parsing.")
                    research_plan = response.strip()
            
            except Exception as e:
                logging.exception("Parsing failed. Using raw response.")
                research_plan = response.strip()
            
            # Final fallback if queries empty
            if not search_queries:
                logging.warning("No queries extracted. Generating fallback query.")
                search_queries = [topic]

            logging.info(
                "Parsed Output | PlanLength=%d | QueriesCount=%d",
                len(research_plan),
                len(search_queries),
            )

            logging.info("ORCHESTRATOR EXECUTION END")

            return {
                "research_plan": research_plan,
                "search_queries": search_queries
            }

        except Exception as e:
            logging.exception("Error during OrchestratorAgent execution.")
            raise CustomException(e, sys)