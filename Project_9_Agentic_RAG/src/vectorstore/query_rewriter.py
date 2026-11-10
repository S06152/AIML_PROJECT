import sys
import re
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.agents.agents import Agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import warnings
warnings.filterwarnings("ignore")

class QueryRewriter:
    """
    Rewrites the user's original question into alternative
    search queries using an LLM.
    """

    def __init__(self, user_controls: dict = None) -> None:

        try:
            self.user_controls = user_controls
            self.llm = Agent()._get_base_llm(self.user_controls)

            logging.info("QueryRewriter initialized successfully.")

        except Exception as e:
            logging.exception("QueryRewriter initialization failed.")
            raise CustomException(e, sys) from e

    def rewrite(self, query: str) -> list[str]:
        """
        Generate alternative search queries.

        The original question is always preserved.
        """

        try:
            if not query or not query.strip():
                return []

            query = query.strip()
            
            message = """
            You are a helpful assistant. Expand the following query to improve document retrieval by adding relevant synonyms, technical terms, and useful context.

            Original query: "{query}"

            Expanded query:
            """

            query_expansion_prompt = ChatPromptTemplate.from_messages(
                [
                    ("user", message)
                ]
            )

            query_expansion_chain = query_expansion_prompt | self.llm | StrOutputParser()
            

            response = query_expansion_chain.invoke({"query": query})

            return response

        except Exception:
            logging.exception(
                "Query rewriting failed. Using the original query."
            )

            return query
