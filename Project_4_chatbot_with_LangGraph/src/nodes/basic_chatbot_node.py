import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.state.state import State

class BasicChatbotNode:
    """
    Basic Chatbot Node implementation.

    This node takes the current conversation state,
    sends it to the LLM, and returns the updated messages.
    """

    def __init__(self, model):
        try:
            logging.info("Initializing BasicChatbotNode")
            self.llm = model

        except Exception as e:
            logging.error("Error during BasicChatbotNode initialization")
            raise CustomException(e, sys)

    def process(self, state: State) -> dict:
        """
        Processes the input state and generates a chatbot response.

        Args:
            state (State): Current graph state containing messages

        Returns:
            dict: Updated state with new messages from LLM
        """
        try:
            logging.info("BasicChatbotNode processing started")

            # Validate input state
            if "messages" not in state:
                logging.warning("State does not contain 'messages' key")
                raise ValueError("State must contain 'messages'")

            messages = state["messages"]
            logging.info(f"Invoking LLM with {len(messages)} message(s)")

            # Call LLM
            response = self.llm.invoke(messages)

            logging.info("LLM response received successfully")

            return {"messages": response}

        except Exception as e:
            logging.error(f"Error in BasicChatbotNode process: {str(e)}")
            raise CustomException(e, sys)