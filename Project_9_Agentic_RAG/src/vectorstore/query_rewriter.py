import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.agents.agents import Agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import warnings
warnings.filterwarnings("ignore")

class QueryRewriter:
    """
    Rewrites the user's original question into an expanded
    search query using an LLM.

    Responsibilities:
        - Initialize the configured LLM.
        - Expand the original query with relevant terminology.
        - Preserve the original query.
        - Return an enhanced query for document retrieval.
    """

    def __init__(self, user_controls: dict = None) -> None:
        """
        Initialize the QueryRewriter.

        Args:
            user_controls (dict): User-selected configuration settings.

        Raises:
            CustomException: If initialization fails.
        """

        try:
            self.user_controls = user_controls

            # Initialize the base LLM
            self.llm = Agent()._get_base_llm(self.user_controls)

            logging.info("QueryRewriter initialized successfully.")

        except Exception as e:
            logging.exception("QueryRewriter initialization failed.")
            raise CustomException(e, sys)

    def rewrite(self, query: str) -> str:
        """
        Expand the original query to improve document retrieval.

        The original question is always preserved.

        Args:
            query (str): Original user question.

        Returns:
            str: Expanded search query.
        """

        try:
            if not query or not query.strip():
                return ""

            query = query.strip()

            logging.info("Original query: %s", query)

            # Query expansion prompt
            message = """
            You are an expert query rewriting assistant.

            Expand the user's question to improve document retrieval.

            Instructions:
            1. Preserve the original meaning.
            2. Add relevant synonyms and technical terminology.
            3. Include useful contextual keywords.
            4. Do not answer the question.
            5. Return only the expanded search query.

            Original query: {query}

            Expanded query:
            """

            query_expansion_prompt = ChatPromptTemplate.from_messages(
                [
                    ("user", message)
                ]
            )

            # LCEL chain
            query_expansion_chain = query_expansion_prompt | self.llm | StrOutputParser()

            # Generate expanded query
            response = query_expansion_chain.invoke({"query" : query})
            enhanced_query = response.strip()

            # Preserve the original query
            if not enhanced_query:
                enhanced_query = query
            elif enhanced_query != query:
                enhanced_query = f"{query} {enhanced_query}"

            logging.info("Enhanced query: %s", enhanced_query)

            return enhanced_query

        except Exception:
            logging.exception("Query rewriting failed. Using the original query.")
            return query