"""
base_agent.py — Abstract base class for all AUTOSAR MAS agents.

All five specialist agents (PM, Architect, Developer, QA, Reviewer)
inherit from BaseAgent and share a common LangChain LCEL pipeline.

Design Pattern: Template Method — BaseAgent defines the pipeline skeleton;
subclasses define the system prompt (role identity) and execute() logic.

MNC Standard 2026:
    - OOP-first: all agents are proper Python classes with docstrings.
    - Single Responsibility: BaseAgent only handles pipeline execution.
    - Open/Closed: new agents can be added by subclassing, not modifying.
"""

import sys
from abc import ABC, abstractmethod
from typing import Any
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.models.state import DevTeamState
from src.utils.logger import logger
from src.utils.exception import CustomException


class BaseAgent(ABC):
    """
    Abstract base class providing common LangChain pipeline functionality
    for all AUTOSAR MAS specialist agents.

    Subclasses must:
        1. Call super().__init__(llm, system_prompt) in __init__.
        2. Implement execute(state) to read/write DevTeamState fields.

    The run() method executes the core LCEL pipeline:
        ChatPromptTemplate → ChatGroq LLM → StrOutputParser

    Args:
        llm           (ChatGroq): Shared Groq LLM instance.
        system_prompt (str)     : Role-defining system instruction.
    """

    def __init__(self, llm: ChatGroq, system_prompt: str) -> None:
        """
        Initialize BaseAgent with LLM and role-defining system prompt.

        Args:
            llm           (ChatGroq): Shared Groq LLM instance.
            system_prompt (str)     : System instruction defining agent's role.

        Raises:
            CustomException: If LLM or system_prompt is invalid.
        """
        try:
            logger.info("Initializing %s.", type(self).__name__)

            if llm is None:
                raise ValueError("LLM instance must not be None.")
            if not system_prompt or not system_prompt.strip():
                raise ValueError("system_prompt must be a non-empty string.")

            self._llm: ChatGroq = llm
            self._system_prompt: str = system_prompt.strip()

            logger.info("%s initialized successfully.", type(self).__name__)

        except Exception as e:
            raise CustomException(e, sys) from e

    def run(self, input_text: str) -> str:
        """
        Execute the LangChain LCEL pipeline for this agent.

        Pipeline:
            ChatPromptTemplate(system + human) → ChatGroq → StrOutputParser

        Args:
            input_text (str): Human message content (artifact from previous agent).

        Returns:
            str: Plain-string LLM response (no markup wrappers).

        Raises:
            CustomException: If prompt construction or LLM invocation fails.
        """
        try:
            logger.info(
                "%s.run() called. Input length: %d chars.",
                type(self).__name__, len(input_text),
            )

            # Build prompt template with system role + human input
            prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    ("system", self._system_prompt),
                    ("human", "{input_text}"),
                ]
            )

            # Compose LCEL pipeline: prompt | LLM | parser
            chain = prompt | self._llm | StrOutputParser()

            # Execute pipeline
            response: str = chain.invoke({"input_text": input_text})

            logger.info(
                "%s.run() completed. Response length: %d chars.",
                type(self).__name__, len(response),
            )
            return response

        except Exception as e:
            raise CustomException(e, sys) from e

    @abstractmethod
    def execute(self, state: DevTeamState) -> dict[str, Any]:
        """
        Agent-specific execution logic. Must be implemented by subclasses.

        Args:
            state (DevTeamState): Shared workflow state dict.

        Returns:
            dict: Partial state update with the agent's output key.
        """
        ...
