from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    """
    Shared state object used throughout the LangGraph workflow.

    Attributes:
        messages: Complete conversation history including
                  HumanMessage, AIMessage, and ToolMessage.
        question: Original user query.
        rewritten_queries: Rewritten or expanded search queries.
        user_controls: User configuration settings (LLM model, API keys, etc.)
    """

    messages: Annotated[List[AnyMessage], add_messages]
    question: str
    rewritten_queries: List[str]
    user_controls: dict