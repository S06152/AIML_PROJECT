import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import warnings
warnings.filterwarnings("ignore")

class BaseAgent:
    """
    Base class for all agents in the Multi-Agent Research & Report Generator.

    Responsibilities:
        - Store shared LLM instance
        - Define agent behavior via system prompt
        - Build reusable LangChain pipeline
        - Execute LLM calls and return parsed output
    """

    def __init__(self, llm: ChatGroq, system_prompt: str) -> None:
        """
        Initialize BaseAgent with LLM and system prompt.

        Args:
            llm (ChatGroq): Shared LLM instance
            system_prompt (str): Role-specific system instruction

        Raises:
            CustomException: If initialization fails
        """

        try:
            logging.info("BASE AGENT INITIALIZATION START")
            logging.info("Initializing BaseAgent | class=%s", type(self).__name__)

            if llm is None:
                raise ValueError("LLM instance must not be None.")

            if not system_prompt or not system_prompt.strip():
                raise ValueError("system_prompt must be a non-empty string.")

            # Initialize LLM 
            self._llm: ChatGroq = llm

            # Store system prompt for later execution
            self._system_prompt: str = system_prompt.strip()

            # Build chain once (performance optimization)
            self._chain = self._build_chain()

            logging.info("BaseAgent initialized successfully | class=%s", type(self).__name__)

            logging.info("BASE AGENT INITIALIZATION COMPLETE")

        except Exception as e:
            logging.exception("ERROR during BaseAgent initialization.")
            raise CustomException(e, sys)

    def _build_chain(self):
        """
        Build LangChain pipeline once and reuse.

        Returns:
            Runnable chain
        """
        try:
            logging.info("Building LLM chain | class = %s", type(self).__name__)

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self._system_prompt),
                    ("human", "{input_text}")
                ]
            )

            chain = prompt | self._llm | StrOutputParser()

            logging.info("LLM chain built successfully | class = %s", type(self).__name__)

            return chain

        except Exception as e:
            logging.exception("Error while building LLM chain")
            raise CustomException(e, sys)
        
    def run(self, input_text: str) -> str:
        """
        Execute LLM pipeline.

        Args:
            input_text (str): Input text (user query or previous agent output)

        Returns:
            str: LLM-generated response

        Raises:
            CustomException: If execution fails
        """

        try:
            logging.info(
                "Agent execution started | Agent = %s | InputLength = %d",
                type(self).__name__,
                len(input_text) if input_text else 0,
            )

            # Execute chain
            response: Optional[str] = self._chain.invoke({"input_text": input_text})

            # Validate Output
            if not response or not response.strip():
                logging.warning("Empty response received from LLM.")
                return ""

            logging.info(
                "Agent execution completed | Agent=%s | OutputLength=%d",
                type(self).__name__,
                len(response),
            )
          
            return response.strip()

        except Exception as e:
            logging.exception("ERROR during agent execution.")
            raise CustomException(e, sys)