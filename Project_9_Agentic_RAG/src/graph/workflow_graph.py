import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from src.agents.agents import Agent
from src.models.state import State
import warnings
warnings.filterwarnings("ignore")

class GraphBuilder:
    """
    Build and execute the Agentic RAG workflow.

    Available tools:
        - vector_db_retriever
        - wikipedia_search
        - arxiv_search
        - tavily_web_search
    """

    def __init__(self) -> None:
        try:
            logging.info("Initializing GraphBuilder.")

            self._tool_call = Agent()
            self._compiled_graph: Optional[Any] = None

            logging.info("GraphBuilder initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize GraphBuilder.")
            raise CustomException(e, sys)

    def build_graph(self):
        """
        Build and compile the LangGraph workflow.
        """

        try:
            if self._compiled_graph is not None:
                logging.info("Returning cached workflow graph.")
                return self._compiled_graph

            logging.info("Building workflow graph.")

            graph = StateGraph(State)

            graph.add_node("tool_calling_llm", self._tool_call.tool_calling_llm)
            graph.add_node("tools", ToolNode(self._tool_call._tools))

            graph.add_edge(START, "tool_calling_llm")
            graph.add_conditional_edges("tool_calling_llm", tools_condition)
            graph.add_edge("tools", "tool_calling_llm")

            self._compiled_graph = graph.compile()

            logging.info("Workflow graph compiled successfully.")

            return self._compiled_graph

        except Exception as e:
            logging.exception("Error while building workflow graph.")
            raise CustomException(e, sys)

    def execute(self,graph,query: str) -> tuple[str, str]:
        """
        Execute the Agentic RAG workflow.

        Args:
            graph: Compiled LangGraph workflow.
            query (str): User query.

        Returns:
            tuple:
                (
                    final_response,
                    tool_name
                )
        """

        try:
            logging.info("Starting workflow execution | Query = '%s'", query )

            initial_state = {
                "messages": [HumanMessage(content=query)],
                "question": query
            }

            final_state = graph.invoke(initial_state)
            messages = final_state.get("messages",[])

            tool_name = self._extract_tool_name(messages)
            response = self._extract_final_response( messages)

            logging.info("Workflow completed successfully. Tool Used: %s", tool_name if tool_name else "None")

            return response, tool_name

        except Exception as e:
            logging.exception("Workflow execution failed.")
            raise CustomException(e, sys)

    @staticmethod
    def _extract_tool_name(messages) -> str:
        """
        Extract tool name from workflow messages.
        """

        for msg in messages:
            if(hasattr(msg, "tool_calls") and msg.tool_calls):
                return msg.tool_calls[0].get("name", "")

            if (hasattr(msg, "type") and msg.type == "tool" and hasattr(msg, "name")):
                return msg.name

        return ""

    @staticmethod
    def _extract_final_response(messages) -> str:
        """
        Extract the final AI response.
        """

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                if (hasattr(msg, "tool_calls")and msg.tool_calls):
                    continue

                return msg.content

        return ""