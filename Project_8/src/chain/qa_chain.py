# Standard Library Imports
import sys
from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.chain.prompt_templates import get_rag_prompt

class QAChain:
    """
    Generic Multi-Modal RAG QA Chain.

    This QA system allows users to upload ANY PDF document:
        - Research Papers
        - Financial Reports
        - Medical PDFs
        - Technical Documents
        - Legal Documents
        - Books
        - Notes
        - Invoices
        - Manuals
        - Academic Papers

    The system retrieves relevant:
        - Text
        - Tables
        - Images
        - Charts
        - Figures

    and generates grounded answers using an LLM.
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        groq_api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        """
        Initialize QAChain.

        Args:
            retriever (VectorStoreRetriever):
                LangChain retriever.

            groq_api_key (str):
                Groq API key.

            model_name (str):
                Groq model name.

            temperature (float):
                Controls creativity.

            max_tokens (int):
                Maximum output tokens.
        """

        try:
            logging.info("Initializing Generic Multi-Modal QAChain.")

            self.retriever = retriever
            self.groq_api_key = groq_api_key
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens

            # Lazy chain initialization
            self._chain = None

            logging.info(
                f"QAChain initialized successfully "
                f"(model={model_name})"
            )

        except Exception as e:
            logging.exception("Error during QAChain initialization." )
            raise CustomException(e, sys)

    # BUILD LLM
    def _build_llm(self) -> ChatGroq:
        """
        Create ChatGroq LLM instance.

        Returns:
            ChatGroq
        """

        try:
            logging.info("Building ChatGroq model." )

            llm = ChatGroq(
                groq_api_key = self.groq_api_key,
                model_name = self.model_name,
                temperature = self.temperature,
                max_tokens = self.max_tokens,
            )

            logging.info("ChatGroq initialized successfully.")

            return llm

        except Exception as e:
            logging.exception("Failed to initialize LLM.")
            raise CustomException(e, sys)

    # FORMAT RETRIEVED DOCUMENTS
    def format_docs(self, docs: List[Document]) -> str:
        """
        Convert retrieved documents into structured context.

        Args:
            docs (List[Document])

        Returns:
            str
        """

        try:
            logging.info(f"Formatting {len(docs)} documents.")

            if not docs:

                return (
                    "No relevant context retrieved "
                    "from the uploaded document."
                )

            formatted_context = []

            for index, doc in enumerate(docs):

                metadata = doc.metadata or {}
                source = metadata.get("source", "Uploaded PDF")
                page = metadata.get("page", "N/A")
                modality = metadata.get("modality", "text")

                formatted_doc = f"""
                    ==================================================
                    DOCUMENT {index + 1}
                    ==================================================
                    Source   : {source}
                    Page     : {page}
                    Modality : {modality}

                    CONTENT:
                    {doc.page_content}
                """

                formatted_context.append(formatted_doc.strip())

            final_context = "\n\n".join(formatted_context )

            logging.info("Document formatting completed.")

            return final_context

        except Exception as e:
            logging.exception("Error while formatting documents.")
            raise CustomException(e, sys)

    # BUILD RAG CHAIN
    def build_chain(self):
        """
        Build complete RAG pipeline.

        Pipeline:
            User Query
                ↓
            Retriever
                ↓
            Context Formatter
                ↓
            Prompt Template
                ↓
            LLM
                ↓
            Output Parser

        Returns:
            Runnable chain
        """

        try:
            if self._chain is None:
                logging.info("Building RAG chain.")

                llm = self._build_llm()

                prompt = get_rag_prompt()

                # RAG PIPELINE
                self._chain = (
                    {
                        "context": self.retriever | RunnableLambda(self.format_docs),
                        "question": RunnablePassthrough()
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                logging.info("RAG chain built successfully.")

            else:

                logging.info("Using cached RAG chain.")

            return self._chain

        except Exception as e:
            logging.exception("Error while building RAG chain.")
            raise CustomException(e, sys)

    # RUN QUERY
    def run(self, query: str) -> str:
        """
        Execute query on uploaded PDF.

        Workflow:
            1. Retrieve relevant chunks
            2. Ground response using context
            3. Generate final answer

        Args:
            query (str):
                User question.

        Returns:
            str:
                Final grounded response.
        """

        try:
            logging.info(f"Executing user query: {query}")

            if not query.strip():
                raise ValueError("Query cannot be empty.")

            # Build or reuse chain
            chain = self.build_chain()

            # Execute pipeline
            response = chain.invoke(query)

            logging.info( "Query executed successfully.")

            return response

        except Exception as e:
            logging.exception("Error while executing query.")
            raise CustomException(e, sys)