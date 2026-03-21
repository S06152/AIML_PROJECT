# Standard library import
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_core.prompts import ChatPromptTemplate

System_prompt = """You are a helpful assistant that answers questions based on the provided context.
Always cite the source document and page number when available.

Context:
{context}

Instructions:
- Answer the question based ONLY on the provided context.
- If the context does not contain enough information, say "I don't have enough information to answer this question."
- Cite sources in the format [Source: filename, Page: X] when available.
- Be concise and accurate.

Answer:
"""

def get_rag_prompt() -> ChatPromptTemplate:
    """
    Creates and returns the RAG (Retrieval-Augmented Generation) prompt template.

    This function builds a structured chat prompt consisting of:
    - A system message (behavior + instructions)
    - A human message (user query placeholder)

    Returns:
        ChatPromptTemplate: A LangChain prompt template object ready to be used in a QA chain.
    """

    try:
        logging.info("Initializing RAG prompt template.")

        # Create a ChatPromptTemplate using system + human message structure
        # "system" defines model behavior
        # "human" injects the user query dynamically via {question}
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", System_prompt),
                ("human", "{question}")
            ]
        )

        logging.info("RAG prompt template created successfully.")

        return prompt
    
    except Exception as e:
        logging.error("Error occurred while creating RAG prompt template.")
        raise CustomException(e, sys)