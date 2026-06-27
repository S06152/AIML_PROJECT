from typing import Annotated
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
        user_controls: User configuration settings (LLM model, API keys, etc.)
    """

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    user_controls: dict