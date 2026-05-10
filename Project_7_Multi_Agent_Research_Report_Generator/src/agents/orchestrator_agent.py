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
_ORCHESTRATOR_PROMPT = """
You are a Research Planning Agent.

Tasks:
1. Create a clear, multi-paragraph research plan (200-400 words) that lists
   the angles, sub-topics, stakeholders, statistics, history, current state,
   challenges, and future outlook to investigate.
2. Generate EXACTLY 4 diverse, high-quality web search queries that together
   cover: (a) overview/definition, (b) latest news / current state 2025-2026,
   (c) statistics / market data / numbers, (d) challenges, risks or future
   outlook.

Output Format (follow EXACTLY):

Research Plan:
<multi-paragraph plan>

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
            topic = topic[:300]
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

            # Simple Parsing
            if "Search Queries:" in response:

                parts = response.split("Search Queries:")

                # Research Plan
                research_plan = (parts[0].replace("Research Plan:", "").strip())

                # Queries
                query_lines = parts[1].split("\n")

                for line in query_lines:

                    line = line.replace("-", "").strip()

                    if line and line not in search_queries:
                        search_queries.append(line[:100])

            # Fallback
            if not research_plan:
                research_plan = f"Research topic: {topic}"

            if not search_queries:
                search_queries = [topic[:80]]

            # Keep only 4 queries
            search_queries = search_queries[:4]

            logging.info(
                "Plan Length=%d | Queries=%d",
                len(research_plan),
                len(search_queries)
            )

            logging.info("ORCHESTRATOR EXECUTION END")

            return {
                "research_plan": research_plan[:2000],
                "search_queries": search_queries
            }
           
        except Exception as e:
            logging.exception("Error during OrchestratorAgent execution.")
            raise CustomException(e, sys)