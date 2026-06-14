import sys
import streamlit as st
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.models.state import State
from langchain_groq import ChatGroq
from src.tools.tool_registry import ToolRegistry
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT = """
You are an intelligent Agentic RAG assistant.

Tool Usage Rules:

1. Use vector_db_retriever for uploaded PDF documents,
   private knowledge, and document-specific questions.

2. Use arxiv_search for:
   - research papers
   - academic literature
   - scientific publications

3. Use wikipedia_search for:
   - general knowledge
   - concepts
   - people
   - places
   - historical information

4. Use tavily_web_search for:
   - current events
   - recent developments
   - latest information
   - web information

If a tool is required, call the most appropriate tool.

After receiving tool results:
- Analyze the information.
- Generate a concise and accurate answer.
- Do not simply copy the tool output.
"""

class Agent:

    def __init__(self):
        """
        Initialize Agent with cached tools.
        Tools are created once and reused across invocations.
        """
        self._tools = ToolRegistry.get_tools()
        self._llm_with_tools = None

    def _get_llm_with_tools(self):
        """
        Build (or reuse) the LLM with bound tools.
        Rebuilds only if user configuration has changed.
        """
        user_controls = st.session_state.get("user_controls", {})

        groq_api_key = user_controls["GROQ_API_KEY"] or st.secrets.get("GROQ_API_KEY")
        model_name   = user_controls["LLM_MODEL"]
        temperature  = user_controls["TEMPERATURE"]
        max_tokens   = user_controls["TOKEN"]

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set. Please configure it in Streamlit secrets.")

        llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        # Full toolset (with retriever) — used when documents are uploaded
        self._llm_with_tools = llm.bind_tools(
                self._tools,
                parallel_tool_calls=False,
                tool_choice="auto"
            )

        return self._llm_with_tools


    def tool_calling_llm(self, state: State):
        """
        LangGraph node — the central decision-making node.

        Builds a ChatGroq LLM with all 4 tools bound to it:
            - vector_db_retriever  : domain-specific PDF content (only when docs uploaded)
            - tavily_web_search    : live web / current events
            - wikipedia_search     : encyclopaedic facts
            - arxiv_search         : academic / research papers

        The LLM receives the full message history and decides:
            - Which tool to call (if any), OR
            - Respond directly without a tool call.
        """
        try:
            llm_with_tools = self._get_llm_with_tools()

            messages  = state["messages"]

            if not messages:
                messages = [HumanMessage(content = state["question"])]   

            logging.info("tool_calling_llm: invoking LLM with bound tools.")
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

            response = llm_with_tools.invoke(messages)

            return {"messages": [response]}

        except Exception as e:
            logging.exception("ERROR during workflow execution.")
            raise CustomException(e, sys)
