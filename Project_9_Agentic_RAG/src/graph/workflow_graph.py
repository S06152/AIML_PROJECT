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
                logging.info("Returning cached compiled workflow graph.")
                return self._compiled_graph
            
            logging.info("Building workflow graph...")

            # Create Graph
            graph = StateGraph(State)

            # Add Nodes (Agents)
            # ToolNode uses ALL tools — the LLM decides which subset is available
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
    
    def execute(self, graph, query: str):
        """
        Execute the full multi-agent pipeline.

        Args:
            query (str): User input query

        Returns:
            tuple: (response_str, tool_name, tool_metadata)
                - response_str  : final answer string
                - tool_name     : name of the tool invoked (or empty string)
                - tool_metadata : dict with source info extracted from tool output
                    For vector_db_retriever → {"sources": [{"doc": ..., "page": ...}, ...]}
                    For web tools           → {"urls": [...]}
                    Empty dict if no tool was called.
        """

        try:
            logging.info("WORKFLOW EXECUTION START")
            logging.info(f"User topic: {query}")

            # Initial state
            initial_state = {
                "messages": [HumanMessage(content = query)],
                "question": query,
                "tool_used":  ""
            }

            logging.info("Invoking workflow graph...")
            final_state = graph.invoke(initial_state)
            messages = final_state.get("messages", [])

            # Extract the tool name from the message history
            tool_name = self._extract_tool_name(messages)

            # Extract source metadata from tool output
            tool_metadata = self._extract_tool_metadata(messages, tool_name)

            # Extract the content of the last AI message as the final response
            response = ""
            if messages:
                last_message = messages[-1]
                response = last_message.content if hasattr(last_message, "content") else str(last_message)

            logging.info(f"Tool used: {tool_name if tool_name else 'None (direct response)'}")
            logging.info("Workflow executed successfully.")
            logging.info("WORKFLOW EXECUTION END")

            return response, tool_name, tool_metadata

        except Exception as e:
            logging.exception("ERROR during workflow execution.")
            raise CustomException(e, sys)

    def _extract_tool_name(self, messages) -> str:
        """
        Extract the tool name invoked during the workflow from messages.

        Iterates through the message history and identifies the tool call
        made by the LLM (from AIMessage.tool_calls or ToolMessage.name).

        Args:
            messages (list): List of messages from the workflow execution.

        Returns:
            str: Name of the tool invoked, or empty string if no tool was called.
        """
        tool_name = ""
        for msg in messages:
            # Check AIMessage for tool_calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_name = msg.tool_calls[0].get("name", "")
                break
            # Fallback: check ToolMessage
            if hasattr(msg, "name") and msg.type == "tool":
                tool_name = msg.name
                break
        return tool_name

    def _extract_tool_metadata(self, messages, tool_name: str) -> dict:
        """
        Extract source metadata from ToolMessage content based on which tool was used.

        For vector_db_retriever:
            Parses lines like "[Source: filename.pdf, Page: 3]" from the tool output
            and returns a deduplicated list of {doc, page} dicts.

        For tavily_web_search / arxiv_search / wikipedia_search:
            Parses lines like "Source: https://..." from the tool output and
            returns a deduplicated list of URL strings.

        Args:
            messages (list): Full message history from the workflow.
            tool_name (str): Name of the tool that was called.

        Returns:
            dict: {"sources": [...]} for retriever, {"urls": [...]} for web tools,
                  or {} if no tool was called.
        """
        if not tool_name:
            return {}

        # Find the ToolMessage (the raw tool output)
        tool_content = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                tool_content = msg.content if hasattr(msg, "content") else ""
                break

        if not tool_content:
            return {}

        try:
            if tool_name == "vector_db_retriever":
                # Parse "[Source: filename.pdf, Page: 3]" lines
                import re
                pattern = r"\[Source:\s*(.+?),\s*Page:\s*(\d+)\]"
                matches = re.findall(pattern, tool_content)
                seen = set()
                sources = []
                for doc, page in matches:
                    key = (doc.strip(), page.strip())
                    if key not in seen:
                        seen.add(key)
                        sources.append({"doc": doc.strip(), "page": page.strip()})
                return {"sources": sources}

            else:
                # Parse "Source: <url>" lines produced by tavily / arxiv / wiki tools
                import re
                pattern = r"Source:\s*(https?://\S+)"
                urls = re.findall(pattern, tool_content)
                # Deduplicate while preserving order
                seen = set()
                unique_urls = []
                for url in urls:
                    if url not in seen:
                        seen.add(url)
                        unique_urls.append(url)
                return {"urls": unique_urls}

        except Exception as e:
            logging.warning(f"Could not extract tool metadata: {e}")
            return {}