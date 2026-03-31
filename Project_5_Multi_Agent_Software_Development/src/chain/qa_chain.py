# Standard library import
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List, Optional
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.vectorstores import VectorStoreRetriever
from src.chain.prompt_templates import get_autosar_rag_prompt
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import warnings
warnings.filterwarnings("ignore")

class QAChain:
    """
    LCEL-based Retrieval-Augmented Generation (RAG) chain for AUTOSAR SWS Q&A.

    Pipeline:
        User Question
            → VectorStoreRetriever (top-K AUTOSAR SWS chunks)
            → format_docs() (chunks → context string)
            → ChatPromptTemplate (context + question → prompt)
            → ChatGroq LLM
            → StrOutputParser
            → Answer string

    Design:
        - Lazy chain construction: built on first call to run() or build_chain().
        - Cached chain (_chain): subsequent calls reuse the same pipeline object.
        - Separation of concerns: prompt logic lives in prompt_templates.py.

    Usage:
        chain = QAChain(retriever, groq_api_key="...", model_name="llama-3.3-70b-versatile")
        answer = chain.run("What does SWS_NvM_00001 specify?")
    """

    def __init__(self,
                 retriever: VectorStoreRetriever,
                 groq_api_key: str,
                 model_name: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.2,
                 max_tokens: int = 2000)-> None:
        """
        Initialize QAChain with retriever and LLM configuration.

        Args:
            retriever    (VectorStoreRetriever): Configured AUTOSAR SWS retriever.
            groq_api_key (str)                : Groq Cloud API key.
            model_name   (str)                : Groq model identifier.
            temperature  (float)              : Sampling temperature (0.0–1.0).
            max_tokens   (int)                : Maximum response tokens.

        Raises:
            Raises Exception: If required parameters are missing.
        """
        try:
            logging.info(
                "Initializing QAChain: model='%s', temperature=%.2f, max_tokens=%d.",
                model_name, temperature, max_tokens,
            )

            if not groq_api_key:
                raise ValueError("groq_api_key must not be empty.")
            if retriever is None:
                raise ValueError("retriever must not be None.")

            # Store retriever and LLM configuration
            self._retriever: VectorStoreRetriever = retriever
            self._groq_api_key: str = groq_api_key
            self._model_name: str = model_name
            self._temperature: float = temperature
            self._max_tokens: int = max_tokens

            # Lazy initialization of the chain (built only once)
            self._chain: Optional[object] = None

            logging.info("QAChain initialized successfully.")

        except Exception as e:
            logging.exception("Error during QAChain initialization.")
            raise CustomException(e, sys)

    def _build_llm(self) -> ChatGroq:
        """
        Construct and return a configured ChatGroq LLM instance.

        Returns:
            ChatGroq: Configured Groq LLM.

        Raises:
            Raises Exception: If LLM initialization fails.
        """
        try:
            logging.info("Building ChatGroq LLM for QAChain: model='%s'.", self._model_name)

            llm = ChatGroq(
                api_key = self._groq_api_key,
                model_name = self._model_name,
                temperature = self._temperature,
                max_tokens = self._max_tokens
            )

            logging.info("ChatGroq LLM instance created successfully.")
            return llm

        except Exception as e:
            logging.exception("Failed to build ChatGroq LLM.")
            raise CustomException(e, sys)

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """
        Concatenate retrieved Document page_content into a single context string.

        Args:
            docs (List[Document]): Retrieved AUTOSAR SWS document chunks.

        Returns:
            str: Newline-separated context string for the prompt template.
        """
        try:
            logging.debug(f"Formatting {len(docs)} retrieved documents.")

            # Combine document contents separated by double newline
            return "\n\n".join(doc.page_content for doc in docs)

        except Exception as e:
            logging.exception("Error while formatting documents.")
            raise CustomException(e, sys)

    def build_chain(self):
        """
        Build and cache the LCEL RAG pipeline (lazy singleton).

        Chain composition:
            {
                "context":  retriever | format_docs,
                "question": passthrough
            }
            | prompt_template
            | llm
            | StrOutputParser

        Returns:
            Runnable: Compiled LangChain LCEL chain.

        Raises:
            AutosarMASException: If chain construction fails.
        """
        try:
            # Build chain only once (caching for performance)
            if self._chain is None:
                logging.info("Building RetrievalQA chain.")

                llm: ChatGroq = self._build_llm()
                prompt = get_autosar_rag_prompt()

                # LangChain Runnable composition pipeline
                self._chain = (
                    {
                        # Retrieve documents and format into context string
                        "context": self._retriever | RunnableLambda(self._format_docs),

                        # Pass user query directly
                        "question": RunnablePassthrough()
                    }
                    # Inject context + question into prompt template
                    | prompt

                    # Send formatted prompt to LLM
                    | llm

                    # Parse LLM response into plain string
                    | StrOutputParser()
                )

                logging.info("RetrievalQA chain built successfully.")

            else:
                # Reuse previously built chain
                logging.info("Using cached RetrievalQA chain.")

            return self._chain

        except Exception as e:
            logging.exception("Error while building RetrievalQA chain.")
            raise CustomException(e, sys)

    def run(self, query: str) -> str:
        """
        Execute the RAG pipeline for a given AUTOSAR SWS question.

        Args:
            query (str): User's natural-language question about the SWS document.

        Returns:
            str: Grounded, SWS-cited answer from the LLM.

        Raises:
            AutosarMASException: If chain execution fails.
        """
        try:
            logging.info("QAChain.run() called. Query length: %d chars.", len(query))

            # Build (or reuse) the chain
            chain = self.build_chain()

            # Execute full RAG pipeline
            response: str = chain.invoke(query)

            logging.info("QAChain.run() completed. Response length: %d chars.", len(response))

            return response

        except Exception as e:
            logging.exception("Error while executing query.")
            raise CustomException(e, sys)
