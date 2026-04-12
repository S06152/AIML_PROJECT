"""
qa_chain.py — LCEL-based RAG chain for AUTOSAR SWS question answering.

Combines a VectorStoreRetriever (AUTOSAR SWS chunks) with a ChatGroq LLM
to answer user questions grounded in the uploaded specification document.

Pipeline:
    User Question
      → VectorStoreRetriever (top-K AUTOSAR SWS chunks)
      → format_docs()            (List[Document] → single context string)
      → ChatPromptTemplate       (context + question → formatted prompt)
      → ChatGroq LLM             (LLM inference)
      → StrOutputParser          (AIMessage → plain string)
      → Answer string

Design:
    - Lazy chain construction: built on first call, cached thereafter.
    - Separation of concerns: prompt logic lives in prompt_templates.py.
    - All config validated at __init__ time, not at run time.
"""

import sys
from typing import List, Optional
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.chain.prompt_templates import get_autosar_rag_prompt
from src.utils.logger import logger
from src.utils.exception import CustomException


class QAChain:
    """
    LCEL Retrieval-Augmented Generation (RAG) chain for AUTOSAR SWS Q&A.

    Usage:
        chain = QAChain(
            retriever=vector_retriever,
            groq_api_key="gsk_...",
            model_name="llama-3.3-70b-versatile",
        )
        answer = chain.run("What does SWS_Com_00001 specify?")
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        groq_api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        """
        Initialize QAChain with retriever and LLM configuration.

        Args:
            retriever    (VectorStoreRetriever): Configured AUTOSAR SWS retriever.
            groq_api_key (str)                : Groq Cloud API key.
            model_name   (str)                : Groq model identifier.
            temperature  (float)              : Sampling temperature (0.0–1.0).
            max_tokens   (int)                : Maximum output tokens.

        Raises:
            CustomException: If required parameters are missing or invalid.
        """
        try:
            logger.info(
                "Initializing QAChain: model='%s', temperature=%.2f, max_tokens=%d.",
                model_name, temperature, max_tokens,
            )

            if not groq_api_key:
                raise ValueError("groq_api_key must not be empty.")
            if retriever is None:
                raise ValueError("retriever must not be None.")

            self._retriever: VectorStoreRetriever = retriever
            self._groq_api_key: str   = groq_api_key
            self._model_name: str     = model_name
            self._temperature: float  = temperature
            self._max_tokens: int     = max_tokens
            self._chain: Optional[object] = None  # Lazy singleton

            logger.info("QAChain initialized successfully.")

        except Exception as e:
            raise CustomException(e, sys) from e

    def _build_llm(self) -> ChatGroq:
        """
        Construct a configured ChatGroq LLM instance.

        Returns:
            ChatGroq: Ready-to-use LLM instance.
        """
        try:
            logger.info("Building ChatGroq LLM for QAChain.")
            llm = ChatGroq(
                api_key=self._groq_api_key,
                model_name=self._model_name,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            logger.info("ChatGroq LLM for QAChain created.")
            return llm

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """
        Join retrieved Document page_content into a single context string.

        Args:
            docs (List[Document]): Retrieved AUTOSAR SWS chunks.

        Returns:
            str: Double-newline separated context string.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def build_chain(self):
        """
        Build and cache the LCEL RAG pipeline (lazy singleton).

        Chain composition:
            {
                "context":  retriever | format_docs,
                "question": passthrough
            }
            | prompt_template | llm | StrOutputParser

        Returns:
            Runnable: Compiled LangChain LCEL chain.
        """
        try:
            if self._chain is None:
                logger.info("Building LCEL RAG chain.")

                llm    = self._build_llm()
                prompt = get_autosar_rag_prompt()

                self._chain = (
                    {
                        "context":  self._retriever | RunnableLambda(self._format_docs),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                logger.info("LCEL RAG chain built and cached.")
            else:
                logger.info("Reusing cached LCEL RAG chain.")

            return self._chain

        except Exception as e:
            raise CustomException(e, sys) from e

    def run(self, query: str) -> str:
        """
        Execute the RAG pipeline for a given AUTOSAR SWS question.

        Args:
            query (str): Natural-language question about the SWS document.

        Returns:
            str: Grounded, SWS-cited answer from the LLM.

        Raises:
            CustomException: If chain execution fails.
        """
        try:
            logger.info("QAChain.run() called. Query length: %d chars.", len(query))

            chain    = self.build_chain()
            response = chain.invoke(query)

            logger.info("QAChain.run() completed. Response: %d chars.", len(response))
            return response

        except Exception as e:
            raise CustomException(e, sys) from e
