import sys
import streamlit as st
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.models.state import State
from langchain_groq import ChatGroq
from src.tools.tool_registry import ToolRegistry


class Agent:

    def tool_calling_llm(state: State):
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
            # Retrieve user controls from session state (set by Streamlit UI)
            user_controls = st.session_state.get("user_controls", {})

            groq_api_key = user_controls.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
            model_name   = user_controls.get("LLM_MODEL", "llama-3.3-70b-versatile")
            temperature  = user_controls.get("TEMPERATURE", 0.2)
            max_tokens   = user_controls.get("TOKEN", 800)

            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is not set. Please configure it in Streamlit secrets.")

            # Build the LLM and bind all tools so it can decide which one to call
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )

            tools = ToolRegistry().get_tools()
            llm_with_tools = llm.bind_tools(tools)   # ← LLM can now select any of the 4 tools

            logging.info("tool_calling_llm: invoking LLM with bound tools.")
            response = llm_with_tools.invoke(state["messages"])

            return {"messages": [response]}

        except Exception as e:
            logging.exception("Error in tool_calling_llm node.")
            raise CustomException(e, sys)
