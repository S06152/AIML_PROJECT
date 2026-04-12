"""
prompt_templates.py — Centralized prompt templates for the AUTOSAR MAS RAG chain.

Defines the system prompt and ChatPromptTemplate used by QAChain for
grounded question-answering over uploaded AUTOSAR SWS documents.

Design: Pure functions — no state, no side effects, easy to test.
"""

import sys
from langchain_core.prompts import ChatPromptTemplate
from src.utils.logger import logger
from src.utils.exception import CustomException

# ---------------------------------------------------------------------------
# AUTOSAR SWS RAG System Prompt
# ---------------------------------------------------------------------------
_AUTOSAR_RAG_SYSTEM_PROMPT: str = """
You are an expert AUTOSAR technical consultant with deep knowledge of
AUTOSAR Classic Platform BSW specifications (SWS documents).

You answer questions strictly based on the AUTOSAR SWS document context provided below.

## Context from AUTOSAR SWS Document:
{context}

## Instructions:
- Answer ONLY based on the provided context. Do not use external knowledge.
- Always cite the relevant SWS clause number if visible in the context
  (e.g., [SWS_Com_00001]).
- If the context does not contain sufficient information, state:
  "The uploaded AUTOSAR SWS document does not contain sufficient information
   to answer this question."
- Use precise AUTOSAR terminology.
- Format answers clearly using bullet points or numbered lists where appropriate.
- Be concise and technically accurate.

## Answer:
"""


def get_autosar_rag_prompt() -> ChatPromptTemplate:
    """
    Build and return the AUTOSAR SWS RAG ChatPromptTemplate.

    Template variables:
        - {context}  : Retrieved AUTOSAR SWS chunks (injected by retriever).
        - {question} : User's question about the AUTOSAR SWS document.

    Returns:
        ChatPromptTemplate: Compiled prompt ready for the QAChain pipeline.

    Raises:
        CustomException: If template construction fails.
    """
    try:
        logger.info("Building AUTOSAR SWS RAG prompt template.")

        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                ("system", _AUTOSAR_RAG_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

        logger.info("AUTOSAR RAG prompt template built successfully.")
        return prompt

    except Exception as e:
        raise CustomException(e, sys) from e
