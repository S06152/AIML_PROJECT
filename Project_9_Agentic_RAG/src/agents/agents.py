import sys
import streamlit as st
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.models.state import State
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage
from src.tools.tool_registry import ToolRegistry
from src.chain.prompt_templates import System_prompt as RESPONSE_FORMAT_PROMPT

# Routing prompt when NO documents are uploaded
TOOL_USE_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

ROUTING RULES — follow them strictly based on the user's question:
1. Current events, recent news, live/real-time information → use the web search tool.
2. Academic research, scientific papers, arXiv topics → use the arXiv search tool.
3. General knowledge, definitions, history, famous people, encyclopaedic facts → use the Wikipedia tool.
4. If you are 100% certain of the answer and it is a simple/trivial question (e.g., "what is 2+2?") → respond directly without any tool.

IMPORTANT: Do NOT use the document retriever tool unless the user has uploaded documents. If no documents are uploaded, use Wikipedia or web search for general questions."""

# Routing prompt when documents ARE uploaded
TOOL_USE_SYSTEM_PROMPT_WITH_DOCS = """You are a helpful AI assistant with access to tools.

The user has uploaded and indexed documents. Follow these routing rules:

1. If the user explicitly asks to "search the web" or about "latest news" or "current events" → use the web search tool.
2. If the user explicitly asks to "search Wikipedia" → use the Wikipedia tool.
3. If the user explicitly asks to "search arXiv" or "find research papers" → use the arXiv tool.
4. For ALL other questions → ALWAYS use the document retriever tool to search the uploaded documents first.

You MUST use the document retriever for any question that could potentially be answered from the uploaded documents. Do NOT answer from your own knowledge."""


class Agent:

    def __init__(self):
        """
        Initialize Agent with cached tools.
        Tools are created once and reused across invocations.
        """
        self._tools = ToolRegistry.get_tools()
        self._llm_with_tools = None
        self._llm_with_tools_no_retriever = None
        self._last_config = None

    def _get_llm_with_tools(self):
        """
        Build (or reuse) the LLM with bound tools.
        Rebuilds only if user configuration has changed.
        
        When documents are uploaded → bind ALL tools (including retriever).
        When NO documents uploaded → bind only web/wiki/arxiv tools (exclude retriever).
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
            # Full toolset (with retriever) — used when documents are uploaded
            self._llm_with_tools = llm.bind_tools(
                self._tools,
                parallel_tool_calls=False,
                tool_choice="auto"
            )
            # Reduced toolset (without retriever) — used when NO documents uploaded
            tools_no_retriever = [t for t in self._tools if t.name != "vector_db_retriever"]
            self._llm_with_tools_no_retriever = llm.bind_tools(
                tools_no_retriever,
                parallel_tool_calls=False,
                tool_choice="auto"
            )
            self._last_config = current_config

        # Return the appropriate LLM based on whether docs are uploaded
        if st.session_state.get("vector_retriever") is not None:
            return self._llm_with_tools
        else:
            return self._llm_with_tools_no_retriever

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

            messages = state["messages"]

            # Detect if tool results are already in the conversation
            # If yes → this is the 2nd call (response generation), use formatting prompt
            # If no  → this is the 1st call (tool routing), use routing prompt
            has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)

            if has_tool_results:
                # 2nd invocation: Tool returned results → use formatting prompt for structured answer
                system_content = RESPONSE_FORMAT_PROMPT
                logging.info("tool_calling_llm: Using response formatting prompt (post-tool).")
            elif st.session_state.get("vector_retriever") is not None:
                # 1st invocation with docs uploaded: Force retriever tool
                system_content = TOOL_USE_SYSTEM_PROMPT_WITH_DOCS
                logging.info("tool_calling_llm: Using document retriever routing prompt.")
            else:
                # 1st invocation without docs: General routing (retriever excluded from tools)
                system_content = TOOL_USE_SYSTEM_PROMPT
                logging.info("tool_calling_llm: Using general routing prompt (no docs).")

            # Prepend/replace system message
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_content)] + list(messages)
            else:
                messages = [SystemMessage(content=system_content)] + list(messages[1:])

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
