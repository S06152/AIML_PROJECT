import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Optional, Any
from src.models.state import State
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from src.agents.agents import Agent
import warnings
warnings.filterwarnings("ignore")

class GraphBuilder:
    """
    1. vector_db_retriever   — proprietary / domain-specific internal documents
    2. tavily_web_search     — current events, recent news, live web information
    3. wikipedia_search      — encyclopaedic facts, definitions, historical info
    4. arxiv_search          — scientific research papers, academic findings
    """

    def __init__(self) -> None:
        """
        Initialize LLM and all agents.

        Args:
            user_controls_input (dict): UI configuration

        Raises:
            CustomException: If initialization fails
        """

        try:
            logging.info("GRAPH BUILDER INITIALIZATION START")

            self._tool_call = Agent()
            # Compiled graph cache
            self._compiled_graph: Optional[Any] = None

            logging.info("All agents initialized successfully.")

        except Exception as e:
            logging.exception("ERROR during GraphBuilder initialization.")
            raise CustomException(e, sys)

    def build_graph(self):
        
        try:
            
            if self._compiled_graph is not None:
                logging.info("Returning cached compiled workflow  graph.")
                return self._compiled_graph
            
            logging.info("Building workflow graph...")

            # Create Graph
            graph = StateGraph(State)

            # Add Nodes (Agents)
            graph.add_node("tool_calling_llm", self._tool_call.tool_calling_llm)
            graph.add_node("tools", ToolNode(self._tool_call._tools))
            
            # Define Edges
            graph.add_edge(START, "tool_calling_llm")

            graph.add_conditional_edges(
                "tool_calling_llm",
                # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
                # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
                tools_condition  # routes to tools or END automatically
            )

            graph.add_edge("tools", "tool_calling_llm")
        

            logging.info("Workflow edges defined successfully.")

            # Compile Graph
            self._compiled_graph = graph.compile()
            logging.info("Workflow graph compiled successfully.")

            return self._compiled_graph

        except Exception as e:
            logging.exception("ERROR while building workflow graph.")
            raise CustomException(e, sys)
    
    def execute(self, graph, question: str):
        """
        Execute the full multi-agent pipeline.

        Args:
            topic (str): User input topic

        Returns:
            Dict[str, Any]: Final workflow state
        """

        try:
            logging.info("WORKFLOW EXECUTION START")
            logging.info(f"User topic: {question}")

            # Initial state
            initial_state = {
                "messages": [HumanMessage(content = question)],
                "question": question,
                "tool_used":  ""
            }

            logging.info("Invoking workflow graph...")
            final_state = graph.invoke(initial_state)
            messages = final_state.get("messages", [])

            # Extract the content of the last AI message as the final response
            response = ""
            if messages:
                last_message = messages[-1]
                response = last_message.content if hasattr(last_message, "content") else str(last_message)

            logging.info("Workflow executed successfully.")
            logging.info("WORKFLOW EXECUTION END")

            # Return ONLY essential outputs
            return response

        except Exception as e:
            logging.exception("ERROR during workflow execution.")
            raise CustomException(e, sys)