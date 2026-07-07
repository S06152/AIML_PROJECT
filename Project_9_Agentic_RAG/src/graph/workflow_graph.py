import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
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

    Conversational memory:
        The compiled graph is wired with a LangGraph checkpointer
        (`MemorySaver`), so the `messages` state is automatically
        persisted and reloaded per `thread_id` across calls to
        `execute()`. This gives the agent short-term, thread-scoped
        memory -- follow-up questions in the same session now see
        prior turns, instead of every query starting from a blank
        slate.

        `MemorySaver` keeps this state in-process (RAM), which is the
        right choice for a single-process demo/interview project. For
        a production deployment running multiple workers/replicas, or
        one that needs to survive a restart, swap it for a persistent
        checkpointer with the same interface, e.g.:

            from langgraph.checkpoint.sqlite import SqliteSaver
            checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

        or a Postgres-/Redis-backed checkpointer. No other code in
        this class needs to change -- `build_graph()` and `execute()`
        only depend on the checkpointer interface, not the backend.
    """

    def __init__(self) -> None:
        try:
            logging.info("Initializing GraphBuilder.")

            self._tool_call = Agent()
            self._compiled_graph: Optional[Any] = None

            # Single shared checkpointer instance for the lifetime of
            # this GraphBuilder (and therefore of the FastAPI process).
            # All conversation threads (one per session_id) live inside
            # this one MemorySaver, keyed internally by thread_id.
            self._checkpointer = MemorySaver()

            logging.info("GraphBuilder initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize GraphBuilder.")
            raise CustomException(e, sys)

    def build_graph(self):
        """
        Build and compile the LangGraph workflow.

        The graph is compiled with `checkpointer=self._checkpointer`,
        which enables LangGraph's built-in thread-scoped memory: state
        (here, `messages`) persists between `execute()` calls that
        share the same `thread_id`, and is isolated between calls that
        use different `thread_id`s.
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

            self._compiled_graph = graph.compile(checkpointer = self._checkpointer)

            logging.info("Workflow graph compiled successfully (with checkpointer).")

            return self._compiled_graph

        except Exception as e:
            logging.exception("Error while building workflow graph.")
            raise CustomException(e, sys)

    def execute(
        self,
        graph,
        query: str,
        user_controls: dict = None,
        thread_id: str = "default",
    ) -> tuple[str, str]:
        """
        Execute the Agentic RAG workflow for a single conversational turn.

        Args:
            graph: Compiled LangGraph workflow (compiled with a checkpointer).
            query (str): User query for THIS turn only.
            user_controls (dict): User configuration settings.
            thread_id (str): Identifies the conversation. Pass the same
                value (e.g. your session_id) across turns of the same
                conversation so the checkpointer can load prior history;
                pass a different value to start a fresh conversation
                with no memory of any other thread.

        Returns:
            tuple:
                (
                    final_response,
                    tool_name
                )
        """

        try:
            logging.info(
                "Starting workflow execution | thread_id = '%s' | Query = '%s'",
                thread_id, query,
            )

            # NOTE: this is the key change for conversational memory.
            # We no longer build the FULL message history here -- we only
            # pass the NEW human message for this turn. The checkpointer
            # (configured in build_graph) automatically loads whatever
            # prior `messages` exist for this `thread_id` and appends this
            # new one (via the `add_messages` reducer on State.messages),
            # before the graph runs. On the very first turn of a new
            # thread_id, there is no prior state, so this single message
            # is the entire history -- behavior for new conversations is
            # unchanged.
            turn_input = {
                "messages": [HumanMessage(content = query)],
                "question": query,
                "user_controls": user_controls or {}
            }

            config = {"configurable": {"thread_id": thread_id}}

            final_state = graph.invoke(turn_input, config = config)
            messages = final_state.get("messages", [])

            tool_name = self._extract_tool_name(messages)
            response = self._extract_final_response(messages)

            logging.info(
                "Workflow completed successfully | thread_id = '%s' | Tool Used: %s",
                thread_id, tool_name if tool_name else "None",
            )

            return response, tool_name

        except Exception as e:
            logging.exception("Workflow execution failed.")
            raise CustomException(e, sys)

    @staticmethod
    def _extract_tool_name(messages) -> str:
        """
        Extract tool name from workflow messages.

        NOTE: with memory enabled, `messages` now contains the FULL
        conversation history for this thread, not just this turn. We only
        want the tool used during THIS turn, so we scan from the most
        recent human message forward rather than from the start of the
        whole list.
        """

        last_human_idx = 0
        for idx, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                last_human_idx = idx

        for msg in messages[last_human_idx:]:
            if(hasattr(msg, "tool_calls") and msg.tool_calls):
                return msg.tool_calls[0].get("name", "")

            if (hasattr(msg, "type") and msg.type == "tool" and hasattr(msg, "name")):
                return msg.name

        return ""

    @staticmethod
    def _extract_final_response(messages) -> str:
        """
        Extract the final AI response for the most recent turn.

        Unchanged in spirit from before: walks backward from the end of
        `messages` and returns the first AIMessage that is not itself a
        tool call. With memory enabled this still correctly returns only
        the latest answer, since we always want the most recent
        non-tool-call AIMessage regardless of how much history precedes it.
        """

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                if (hasattr(msg, "tool_calls")and msg.tool_calls):
                    continue

                return msg.content

        return ""

    # NOTE: A `_display_graph` Streamlit-rendering method previously lived
    # here. It was unused dead code and, more importantly, invalid: this
    # module's `execute()` runs inside the FastAPI process, which has no
    # Streamlit ScriptRunContext, so any `st.*` call here is unsafe. Graph
    # visualization is already correctly handled via the `/graph` endpoint
    # in `src/ui/fastapi.py`, which the Streamlit frontend fetches and
    # renders for itself (see `src/main.py:_display_graph_in_sidebar`).
