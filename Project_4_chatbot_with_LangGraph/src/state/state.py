from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    """
    Represents the structure of the state used in the LangGraph workflow.

    Attributes:
        messages (List):
            A list of chat messages passed between nodes.

            The `add_messages` annotation ensures that new messages
            are appended to the existing list instead of replacing it,
            enabling conversational memory across the graph.
    """

    messages: Annotated[List, add_messages]