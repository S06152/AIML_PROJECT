import sys
import re
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
You are a Research Planning Agent. You can plan research on ANY topic
(technology, business, science, society, history, sports, finance, etc.).

Tasks:
1. Create a clear, multi-paragraph research plan (200-400 words) that lists
   the angles, sub-topics, stakeholders, statistics, history, current state,
   challenges, and future outlook to investigate.
2. Identify 6-8 REPORT DIMENSIONS - the specific themes / sub-topics that
   the final report MUST cover for THIS topic. These are domain-adaptive:
   pick whatever is most relevant for the given topic. Each dimension is a
   short noun phrase (2-5 words).
3. Generate EXACTLY 6 diverse, high-quality web search queries that together
   cover: overview/definition, latest news / 2025-2026 state, statistics &
   market data, key players or stakeholders, challenges/risks, and future
   outlook. Each query <= 100 chars.

Output Format (follow EXACTLY, including the headings):

Research Plan:
<multi-paragraph plan>

Report Dimensions:
- dimension 1
- dimension 2
- dimension 3
- dimension 4
- dimension 5
- dimension 6

Search Queries:
- query 1
- query 2
- query 3
- query 4
- query 5
- query 6
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
            report_dimensions: List[str] = []
            search_queries: List[str] = []

            # Section-aware parsing using the three required headings
            text = response

            # 1) Research Plan: everything between "Research Plan:" and the
            #    next known heading.
            plan_match = re.search(
                r"Research Plan:\s*(.*?)(?=\n\s*(?:Report Dimensions:|Search Queries:)|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if plan_match:
                research_plan = plan_match.group(1).strip()

            # 2) Report Dimensions: bulleted list
            dim_match = re.search(
                r"Report Dimensions:\s*(.*?)(?=\n\s*Search Queries:|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if dim_match:
                for raw_line in dim_match.group(1).splitlines():
                    line = raw_line.strip().lstrip("-*0123456789.) ").strip()
                    if line and line not in report_dimensions:
                        report_dimensions.append(line[:80])

            # 3) Search Queries: bulleted list
            q_match = re.search(
                r"Search Queries:\s*(.*)$",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if q_match:
                for raw_line in q_match.group(1).splitlines():
                    line = raw_line.strip().lstrip("-*0123456789.) ").strip()
                    if line and line not in search_queries:
                        search_queries.append(line[:100])

            # Fallbacks
            if not research_plan:
                research_plan = f"Research topic: {topic}"

            if not report_dimensions:
                # Generic, topic-agnostic fallback dimensions
                report_dimensions = [
                    "Overview & Definition",
                    "Current State (2025-2026)",
                    "Key Players & Stakeholders",
                    "Statistics & Market Data",
                    "Challenges & Risks",
                    "Future Outlook",
                ]

            if not search_queries:
                search_queries = [topic[:80]]

            # Keep up to 6 of each
            report_dimensions = report_dimensions[:8]
            search_queries = search_queries[:6]

            logging.info(
                "Plan Length=%d | Dimensions=%d | Queries=%d",
                len(research_plan),
                len(report_dimensions),
                len(search_queries),
            )

            logging.info("ORCHESTRATOR EXECUTION END")

            return {
                "research_plan": research_plan[:2000],
                "report_dimensions": report_dimensions,
                "search_queries": search_queries,
            }
           
        except Exception as e:
            logging.exception("Error during OrchestratorAgent execution.")
            raise CustomException(e, sys)