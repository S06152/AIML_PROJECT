import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Any
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.models.state import DevTeamState

class BaseAgent:
    """
    BaseAgent provides the common functionality required for all AI agents.

    Responsibilities:
    - Initialize the LLM provider
    - Build the prompt template
    - Execute the LangChain pipeline
    - Return the parsed LLM response

    Other agents such as ArchitectAgent, DeveloperAgent, etc.
    should inherit from this class.
    """

    def __init__(self, llm: ChatGroq, system_prompt: str) -> None:
        """
        Initialize BaseAgent with LLM and role-defining system prompt.

        Args:
            llm           (ChatGroq): Shared Groq LLM instance.
            system_prompt (str)     : System instruction defining agent role.

        Raises:
            Raises Exception: If LLM or system_prompt is invalid.
        """

        try:
            logging.info("Initializing BaseAgent: class=%s", type(self).__name__)

            if llm is None:
                raise ValueError("LLM instance must not be None.")

            if not system_prompt or not system_prompt.strip():
                raise ValueError("system_prompt must be a non-empty string.")

            # Initialize LLM 
            self._llm: ChatGroq = llm

            # Store system prompt for later execution
            self._system_prompt: str = system_prompt.strip()

            logging.info("BaseAgent initialized successfully: class=%s", type(self).__name__)

        except Exception as e:
            logging.error("Error while initializing BaseAgent")
            raise CustomException(e, sys)

    def run(self, input_text: str) -> str:
        """
        Execute the LangChain pipeline for this agent.

        Pipeline:
            ChatPromptTemplate(system + human) → ChatGroq LLM → StrOutputParser

        Args:
            input_text (str): The input passed as the human message.
                              Typically an artifact from the previous agent.

        Returns:
            str: Parsed LLM response (plain string, no markup).

        Raises:
            Raises Exception: If prompt building or LLM invocation fails.
        """

        try:
            logging.info(
                "Agent '%s' run() called. Input length: %d chars.",
                type(self).__name__,
                len(input_text),
            )

            # Create prompt template with system and user messages
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self._system_prompt),
                    ("human", "{input_text}")
                ]
            )

            logging.info("Prompt template created successfully")

            # Build LangChain pipeline
            chain = prompt | self._llm | StrOutputParser()

            logging.info("LLM chain constructed")

            # Execute the chain
            response: str = chain.invoke({"input_text": input_text})

            logging.info(
                "Agent '%s' run() completed. Response length: %d chars.",
                type(self).__name__,
                len(response),
            )

            return response

        except Exception as e:
            logging.error("Error occurred during agent execution")
            raise CustomException(e, sys)