# Standard library import
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.vectorstores import VectorStoreRetriever
from src.chain.prompt_templates import get_rag_prompt
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class QAChain:
    """
    Builds a LangChain Retrieval-Augmented Generation (RAG) QA chain
    using:
        - A VectorStore retriever
        - A Groq-hosted LLM
        - A structured prompt template
        - An output parser

    This class encapsulates the full RAG pipeline:
        User Query → Retriever → Context Formatting → Prompt → LLM → Parsed Output
    """

    def __init__(self,
                 retriever: VectorStoreRetriever,
                 groq_api_key: str,
                 model_name: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.2,
                 max_tokens: int = 800):
        """
        Initialize the QAChain with retriever and LLM configuration.

        Parameters:
            retriever (VectorStoreRetriever): Handles document retrieval.
            groq_api_key (str): API key for Groq LLM.
            model_name (str): Model identifier.
            temperature (float): Controls randomness of output.
            max_tokens (int): Maximum tokens in response.
        """
        try:
            logging.info("Initializing QAChain.")

            # Store retriever and LLM configuration
            self.retriever = retriever
            self.groq_api_key = groq_api_key
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens

            # Lazy initialization of the chain (built only once)
            self._chain = None

            logging.info(
                f"QAChain initialized with model = {model_name}, "
                f"temperature = {temperature}, max_tokens = {max_tokens}"
            )

        except Exception as e:
            logging.exception("Error during QAChain initialization.")
            raise CustomException(e, sys)

    def _build_llm(self) -> ChatGroq:
        """
        Create and return the ChatGroq LLM instance.

        This method isolates LLM construction logic
        to keep the class modular and maintainable.
        """
        try:
            logging.info("Building ChatGroq LLM instance.")

            llm = ChatGroq(
                groq_api_key = self.groq_api_key,
                model_name = self.model_name,
                temperature = self.temperature,
                max_tokens = self.max_tokens
            )

            logging.info("ChatGroq LLM instance created successfully.")
            return llm

        except Exception as e:
            logging.exception("Failed to build ChatGroq LLM.")
            raise CustomException(e, sys)

    def format_docs(self, docs: List[Document]) -> str:
        """
        Convert retrieved Document objects into a single formatted string.

        This formatted string becomes the {context}
        injected into the RAG prompt template.
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
        Build and return the RetrievalQA chain.

        The chain architecture:
            {
                "context": retriever → formatter,
                "question": passthrough query
            }
            → prompt template
            → LLM
            → output parser
        """
        try:
            # Build chain only once (caching for performance)
            if self._chain is None:
                logging.info("Building RetrievalQA chain.")

                llm = self._build_llm()
                prompt = get_rag_prompt()

                # LangChain Runnable composition pipeline
                self._chain = (
                    {
                        # Retrieve documents and format into context string
                        "context": self.retriever | RunnableLambda(self.format_docs),

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
        Execute a user query through the RAG pipeline.

        Steps:
            1. Ensure chain is built.
            2. Invoke chain with query.
            3. Return parsed LLM response.
        """
        try:
            logging.info(f"Running query through QAChain: {query}")

            # Build (or reuse) the chain
            chain = self.build_chain()

            # Execute full RAG pipeline
            response = chain.invoke(query)

            logging.info("Query executed successfully.")

            return response

        except Exception as e:
            logging.exception("Error while executing query.")
            raise CustomException(e, sys)
