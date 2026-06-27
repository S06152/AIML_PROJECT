import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.state import State
from src.tools.tool_registry import ToolRegistry
import warnings
warnings.filterwarnings("ignore")

SYSTEM_PROMPT = """
You are an intelligent Agentic RAG assistant.

Your primary responsibility is to understand the user's intent,
select the most appropriate tool when necessary, and provide
accurate, concise, and well-structured answers.

========================
TOOL SELECTION RULES
========================

1. vector_db_retriever
   Use this tool when the user's question relates to:
   - Uploaded PDF documents
   - User-provided files
   - Proprietary or private information
   - Document summaries
   - Questions that should be answered from uploaded content

2. arxiv_search
   Use this tool when the user asks about:
   - Research papers
   - Academic publications
   - Scientific studies
   - Technical literature
   - Scholarly findings
   - State-of-the-art research

3. wikipedia_search
   Use this tool when the user asks about:
   - General knowledge
   - Definitions
   - Concepts
   - Historical events
   - People
   - Places
   - Organizations
   - Background information

4. tavily_web_search
   Use this tool when the user asks about:
   - Current events
   - Recent developments
   - Latest news
   - Live information
   - Information that changes over time
   - Real-time web information

5. Direct LLM Response
   Do NOT call any tool when the question can be answered
   directly using reasoning or general knowledge, such as:
   - Python programming
   - Coding examples
   - Algorithms
   - Mathematics
   - Logical reasoning
   - Writing assistance
   - General explanations

========================
RESPONSE RULES
========================

- Select the single most appropriate tool whenever possible.
- Do not call multiple tools unless absolutely necessary.
- After receiving tool results, analyze and summarize them.
- Do not copy tool output verbatim.
- Generate a clear, accurate, and user-friendly response.
- If no tool is required, answer directly.
- If a tool returns no relevant information, explain that gracefully.
"""

DOCUMENT_CONTEXT_AVAILABLE = """

SESSION CONTEXT:

The user has uploaded and indexed document(s) in this session.

When the question appears related to technical content,
research topics, concepts, methods, findings, architecture,
algorithms, processes, or document content, prefer using
vector_db_retriever first.

If retrieval returns irrelevant or insufficient information,
then use your own knowledge or another suitable tool.

For general knowledge, people, places, organizations,
or current events, use the normal tool-selection rules.
"""

DOCUMENT_CONTEXT_UNAVAILABLE = """

SESSION CONTEXT:

No documents have been uploaded or indexed in this session.

The vector_db_retriever tool is currently unavailable.

Use:
- wikipedia_search
- arxiv_search
- tavily_web_search

or answer directly if no tool is required.
"""

class Agent:
    """
    Central Agent responsible for:
        - Loading tools
        - Configuring the LLM
        - Selecting tools
        - Producing tool calls or direct responses
    """

    def __init__(self) -> None:
        try:
            logging.info("Initializing Agent.")

            self._tools = ToolRegistry.get_tools()

            logging.info("Agent initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize Agent.")
            raise CustomException(e, sys)

    def _get_base_llm(self, user_controls: dict = None):
        """
        Create and return a base ChatGroq instance (without tools).
        """

        try:
            if not user_controls:
                user_controls = st.session_state.get("user_controls", {})
            groq_api_key = user_controls.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
            model_name = user_controls.get("LLM_MODEL")
            temperature = user_controls.get("TEMPERATURE", 0.1)
            max_tokens = user_controls.get("TOKEN", 2048)

            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is not configured.")

            if not model_name:
                raise ValueError("LLM_MODEL is not configured.")

            return ChatGroq(
                groq_api_key = groq_api_key,
                model_name = model_name,
                temperature = temperature,
                max_tokens = max_tokens,
            )

        except Exception as e:
            logging.exception("Failed to initialize base LLM.")
            raise CustomException(e, sys)

    def _get_llm_with_tools(self, user_controls: dict = None):
        """
        Create ChatGroq instance and bind tools.
        """

        try:
            llm = self._get_base_llm(user_controls)

            return llm.bind_tools(
                self._tools,
                parallel_tool_calls = False,
                tool_choice = "auto"
            )

        except Exception as e:
            logging.exception("Failed to initialize LLM with tools.")
            raise CustomException(e, sys)

    def tool_calling_llm(self, state: State):
        """
        LangGraph node responsible for:
            1. Reading user query.
            2. Selecting the best tool.
            3. Returning a tool call or direct answer.
        """

        try:

            user_controls = state.get("user_controls", {})
            llm_with_tools = self._get_llm_with_tools(user_controls)
        
            messages = state.get("messages", [])

            if not messages:
                messages = [HumanMessage(content = state["question"])]

            if st.session_state.get("vector_retriever") is not None:
                system_prompt = SYSTEM_PROMPT + DOCUMENT_CONTEXT_AVAILABLE
            else:
                system_prompt = SYSTEM_PROMPT + DOCUMENT_CONTEXT_UNAVAILABLE

            final_messages = [SystemMessage(content = system_prompt)] + messages

            logging.info("Invoking LLM with %s tool(s).", len(self._tools))

            try:
                response = llm_with_tools.invoke(final_messages)
            except Exception as tool_error:
                error_str = str(tool_error)
                if "tool_use_failed" in error_str or "Failed to call a function" in error_str:
                    logging.warning(
                        "Tool-bound invocation failed (model chose direct response). "
                        "Retrying without tools."
                    )
                    base_llm = self._get_base_llm(user_controls)
                    response = base_llm.invoke(final_messages)
                else:
                    raise

            return {"messages": [response] }

        except Exception as e:
            logging.exception("Error during Agent execution.")
            raise CustomException(e, sys)