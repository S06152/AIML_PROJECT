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

# Appended to SYSTEM_PROMPT when the user HAS uploaded/indexed PDF document(s)
# in this session (i.e. "vector_retriever" is present in st.session_state).
DOCUMENT_CONTEXT_AVAILABLE = """

SESSION CONTEXT:
The user has uploaded and indexed PDF document(s) in this session, so the
vector_db_retriever tool is available. Uploaded documents are typically
research papers, reports, manuals, or other technical/domain-specific
material.

Decide between two categories of question:

1. TECHNICAL / DOCUMENT-LIKELY QUESTIONS — the question asks you to explain,
   define, summarize, or describe a technical term, method, algorithm,
   architecture, process, section, finding, dataset, or other domain-specific
   concept that could plausibly be covered in a research paper or technical
   document — even if the user does not say "in the document" or "according
   to the PDF".
   -> Call vector_db_retriever FIRST, even if you believe you already know the
      answer from general knowledge. If the retrieved content is empty or
      clearly irrelevant, fall back to your own knowledge or another tool.

2. REAL-WORLD ENTITY / GENERAL-KNOWLEDGE QUESTIONS — the question asks "who is
   <person>", or is about a specific real-world person, athlete, celebrity,
   place, organization, current event, or general trivia. These topics are
   essentially never the subject of a technical/research document.
   -> Do NOT call vector_db_retriever for these. Follow the normal Tool Usage
      Rules above instead (wikipedia_search for people/places/general
      knowledge, tavily_web_search for current events, arxiv_search for
      unrelated research topics).

If you are unsure which category applies, prefer the normal Tool Usage Rules
(wikipedia_search / arxiv_search / tavily_web_search) over vector_db_retriever.
"""

# Appended to SYSTEM_PROMPT when NO documents have been uploaded/indexed yet.
DOCUMENT_CONTEXT_UNAVAILABLE = """

SESSION CONTEXT:
No documents have been uploaded in this session, so vector_db_retriever is
NOT available right now. Use wikipedia_search, arxiv_search, or
tavily_web_search as appropriate, or answer directly from your own knowledge
if no tool is needed.
"""

class Agent:

    def __init__(self):
        """
        Initialize Agent with tools.
        Tools are recomputed in tool_calling_llm() on every invocation so
        that a retriever added mid-session (after PDF upload) is picked up
        without needing to recreate the Agent.
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
            # Recompute tools on every call — picks up vector_db_retriever
            # as soon as "vector_retriever" appears in st.session_state,
            # even if it wasn't there when this Agent was constructed
            # (covers: query -> upload PDF -> query again, in the same run).
            self._tools = ToolRegistry.get_tools()
            llm_with_tools = self._get_llm_with_tools()

            messages  = state["messages"]

            if not messages:
                messages = [HumanMessage(content = state["question"])]

            # Tell the LLM whether document retrieval is available right now,
            # so its automatic (tool_choice="auto") routing decision has the
            # context it needs — without forcing any specific tool.
            if "vector_retriever" in st.session_state:
                system_prompt = SYSTEM_PROMPT + DOCUMENT_CONTEXT_AVAILABLE
            else:
                system_prompt = SYSTEM_PROMPT + DOCUMENT_CONTEXT_UNAVAILABLE

            logging.info("tool_calling_llm: invoking LLM with bound tools.")
            messages = [SystemMessage(content=system_prompt)] + messages

            response = llm_with_tools.invoke(messages)

            return {"messages": [response]}

        except Exception as e:
            logging.exception("ERROR during workflow execution.")
            raise CustomException(e, sys)