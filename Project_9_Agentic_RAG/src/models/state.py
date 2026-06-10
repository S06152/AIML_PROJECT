from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

import warnings
warnings.filterwarnings("ignore")

# Graph state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    tool_used: str