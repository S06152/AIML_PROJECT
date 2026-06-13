import sys
import streamlit as st
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.models.state import State
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from src.tools.tool_registry import ToolRegistry

# System message to guide the LLM on proper tool usage
TOOL_USE_SYSTEM_PROMPT = """You are a helpful AI assistant with access to specialized tools.

When you need to use a tool, you MUST use the proper JSON tool calling format provided by the system.
DO NOT use XML-style function calls like <function=name{...}</function>.
Simply decide which tool to use and provide the required arguments.

Available tools:
- vector_db_retriever: Search uploaded PDF documents for domain-specific information
- tavily_web_search: Search the web for current events or real-time information
- wikipedia_search: Search Wikipedia for encyclopaedic facts and general knowledge
- arxiv_search: Search arXiv for academic research papers and scientific publications

If you can answer the question directly without any tool, do so.
If you need information from a tool, call exactly ONE tool at a time."""


class Agent:

    def __init__(self):
        """
        Initialize Agent with cached tools.
        Tools are created once and reused across invocations.
        """
        self._tools = ToolRegistry.get_tools()
        self._llm_with_tools = None
        self._last_config = None

    def _get_llm_with_tools(self):
        """
        Build (or reuse) the LLM with bound tools.
        Rebuilds only if user configuration has changed.
        """
        user_controls = st.session_state.get("user_controls", {})

        groq_api_key = user_controls.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        model_name   = user_controls.get("LLM_MODEL", "llama-3.3-70b-versatile")
        temperature  = user_controls.get("TEMPERATURE", 0.2)
        max_tokens   = user_controls.get("TOKEN", 800)

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set. Please configure it in Streamlit secrets.")

        current_config = (groq_api_key, model_name, temperature, max_tokens)

        # Rebuild LLM only if config has changed
        if self._llm_with_tools is None or self._last_config != current_config:
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            # parallel_tool_calls=False prevents format confusion in Llama models
            self._llm_with_tools = llm.bind_tools(self._tools, parallel_tool_calls=False)
            self._last_config = current_config

        return self._llm_with_tools

    def tool_calling_llm(self, state: State):
        """
        LangGraph node — the central decision-making node.

        Builds a ChatGroq LLM with all 4 tools bound to it:
            - vector_db_retriever  : domain-specific PDF content
            - tavily_web_search    : live web / current events
            - wikipedia_search     : encyclopaedic facts
            - arxiv_search         : academic / research papers

        The LLM receives the full message history and decides:
            - Which tool to call (if any), OR
            - Respond directly without a tool call.
        """
        try:
            llm_with_tools = self._get_llm_with_tools()

            # Prepend system message to guide proper tool calling format
            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=TOOL_USE_SYSTEM_PROMPT)] + list(messages)

            logging.info("tool_calling_llm: invoking LLM with bound tools.")
            response = llm_with_tools.invoke(messages)

            return {"messages": [response]}

        except Exception as e:
            logging.exception("Error in tool_calling_llm node.")
            raise CustomException(e, sys)
