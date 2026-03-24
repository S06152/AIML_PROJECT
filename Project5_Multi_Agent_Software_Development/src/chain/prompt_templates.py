# Standard library import
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# AUTOSAR SWS RAG System Prompt
# ---------------------------------------------------------------------------
_AUTOSAR_RAG_SYSTEM_PROMPT: str = """
You are an expert AUTOSAR technical consultant with deep knowledge of
AUTOSAR Classic Platform BSW specifications (SWS documents).

You answer questions strictly based on the AUTOSAR SWS document context provided.

## Context from AUTOSAR SWS Document:
{context}

## Instructions:
- Answer ONLY based on the provided context. Do not use external knowledge.
- Always cite the relevant SWS clause number if visible in the context
  (e.g., [SWS_NvM_00001]).
- If the context does not contain enough information, respond:
  "The uploaded AUTOSAR SWS document does not contain sufficient information
   to answer this question."
- Use precise AUTOSAR terminology.
- Format answers clearly with bullet points or numbered lists where appropriate.
- Be concise and technically accurate.

## Answer:
"""

def get_autosar_rag_prompt() -> ChatPromptTemplate:
    """
    Build and return the AUTOSAR SWS RAG ChatPromptTemplate.

    Template variables:
        - {context}  : Retrieved AUTOSAR SWS document chunks (injected by retriever).
        - {question} : User's natural-language question about the SWS spec.

    Returns:
        ChatPromptTemplate: Compiled prompt template for the QA chain.

    Raises:
        Raises Exception: If template construction fails.
    """

    try:
        logging.info("Building AUTOSAR SWS RAG prompt template.")

        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                ("system", _AUTOSAR_RAG_SYSTEM_PROMPT),
                ("human", "{question}")
            ]
        )

        logging.info("AUTOSAR SWS RAG prompt template built successfully.")

        return prompt
    
    except Exception as e:
        logging.error("Error occurred while creating RAG prompt template.")
        raise CustomException(e, sys)