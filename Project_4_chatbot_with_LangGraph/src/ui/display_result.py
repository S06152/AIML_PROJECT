import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st

class DisplayResultStreamlit:
    """
    Handles displaying chatbot responses in Streamlit.

    Responsibilities:
    - Maintain chat history using session state
    - Display user & assistant messages
    - Invoke LangGraph and show results
    """

    def __init__(self, graph, user_message):
        try:
            logging.info("Initializing DisplayResultStreamlit")

            self.graph = graph
            self.user_message = user_message

            # Initialize session state for chat history
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
                logging.info("Session state 'messages' initialized")

        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise CustomException(e, sys)

    def display_result(self):
        """
        Handles user input, invokes graph, and displays response.
        """
        try:
            logging.info("Displaying chat history")

            # Step 1: Display existing chat history
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Step 2: Process new user message
            if self.user_message:
                logging.info(f"User message received: {self.user_message}")

                # Store user message
                st.session_state["messages"].append({"role": "user", "content": self.user_message})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(self.user_message)

                response = None

                # Step 4: Generate assistant response
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Generating response..."):
                        try:
                            result = self.graph.invoke({"messages": [("user", self.user_message)]})
                            response = result["messages"][-1].content
                            st.markdown(response)
                            logging.info("Assistant response generated successfully")

                        except Exception as e:
                            logging.error(f"Error during graph invocation: {str(e)}")
                            response = f"❌ Error processing query: {str(e)}"
                            st.error(response)

                # Step 5: Store assistant response
                st.session_state["messages"].append({"role": "assistant", "content": response})

        except Exception as e:
            logging.error(f"Error in display_result: {str(e)}")
            raise CustomException(e, sys)