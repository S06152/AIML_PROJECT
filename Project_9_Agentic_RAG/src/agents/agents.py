import sys
import streamlit as st
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.models.state import State
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from src.tools.tool_registry import ToolRegistry

# Simplified system prompt — avoids listing tool names to prevent format confusion in Llama models.
# Tools are already described to the model via bind_tools(); repeating names here causes XML-style calls.
TOOL_USE_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

IMPORTANT ROUTING RULES — follow them strictly:
1. If the user's question is about content from their uploaded documents, use the document retriever tool.
2. If the user asks about current events or real-time information, use the web search tool.
3. If the user asks about well-known encyclopaedic facts, use the Wikipedia tool.
4. If the user asks about academic research or scientific papers, use the arXiv tool.
5. If you can confidently answer without any tool, respond directly.

Always prefer the document retriever when the user has uploaded documents and asks about their content."""


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
            # tool_choice="auto" ensures proper native tool calling via the API
            # parallel_tool_calls=False prevents format confusion in Llama models
            self._llm_with_tools = llm.bind_tools(
                self._tools,
                parallel_tool_calls=False,
                tool_choice="auto"
            )
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

            # Build system prompt with context about uploaded documents
            system_content = TOOL_USE_SYSTEM_PROMPT
            if st.session_state.get("vector_retriever") is not None:
                system_content += "\n\nNote: The user has uploaded and indexed documents. Prefer searching the uploaded documents first for their questions."

            # Prepend system message to guide proper tool calling format
            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_content)] + list(messages)

            logging.info("tool_calling_llm: invoking LLM with bound tools.")
            response = llm_with_tools.invoke(messages)

            return {"messages": [response]}

        except Exception as e:
            # Handle Groq tool_use_failed error (XML-style generation issue)
            error_str = str(e)
            if "tool_use_failed" in error_str or "Failed to call a function" in error_str:
                logging.warning("LLM generated invalid tool call format. Retrying without system message...")
                try:
                    # Retry with only the user messages (no system prompt that might confuse the model)
                    messages = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]
                    response = llm_with_tools.invoke(messages)
                    return {"messages": [response]}
                except Exception as retry_error:
                    logging.exception("Retry also failed in tool_calling_llm node.")
                    raise CustomException(retry_error, sys)

            logging.exception("Error in tool_calling_llm node.")
            raise CustomException(e, sys)
