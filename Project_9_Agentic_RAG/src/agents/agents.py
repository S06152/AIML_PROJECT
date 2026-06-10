from src.models.state import State
from src.chain.qa_chain import QAChain
class Agent:

    def tool_calling_llm(state: State):
        """
        This node sends messages to the LLM.
        The LLM decides whether to call a tool or respond directly.
        """
        
        return {"messages": [QAChain().run(state["messages"])]}
