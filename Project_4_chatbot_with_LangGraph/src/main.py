import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import streamlit as st
from src.ui.loadui import LoadStreamlitUI
from src.LLM.groq_llm import GROQLLM
from src.graph.graph_builder import GraphBuilder
from src.ui.display_result import DisplayResultStreamlit

class LangGraphApp:
    """
    Main application class for running the LangGraph Agentic AI Streamlit app.

    Responsibilities:
    - Load UI and capture user inputs
    - Initialize LLM model
    - Build LangGraph workflow
    - Execute graph and display results
    """

    def __init__(self):
        try:
            logging.info("Initializing LangGraphApp")

            self.ui = LoadStreamlitUI()
            self.user_input = {}
            self.model = None
            self.graph = None

        except Exception as e:
            logging.error(f"Error during app initialization: {str(e)}")
            raise CustomException(e, sys)

    def _initialize_llm(self):
        """
        Initializes the LLM model using user inputs.
        """
        try:
            logging.info("Initializing LLM model")

            obj_llm_config = GROQLLM(user_contols_input = self.user_input)
            self.model = obj_llm_config.get_llm_model()

            logging.info("LLM model initialized successfully")

        except Exception as e:
            logging.error(f"Error initializing LLM: {str(e)}")
            raise CustomException(e, sys)

    def _build_graph(self):
        """
        Builds the LangGraph workflow.
        """
        try:
            logging.info("Building LangGraph")

            graph_builder = GraphBuilder(self.model)
            self.graph = graph_builder.basic_chatbot_build_graph()

            logging.info("Graph built successfully")

        except Exception as e:
            logging.error(f"Error building graph: {str(e)}")
            raise CustomException(e, sys)

    def _display_result(self, user_message: str):
        """
        Executes the graph and displays the result on UI.
        """
        try:
            logging.info("Displaying result on Streamlit UI")

            DisplayResultStreamlit(self.graph, user_message).display_result_on_ui()

            logging.info("Result displayed successfully")

        except Exception as e:
            logging.error(f"Error displaying result: {str(e)}")
            raise CustomException(e, sys)

    def run(self):
        """
        Entry point to run the Streamlit application.
        """
        try:
            logging.info("Starting LangGraph application")

            # Step 1: Load UI
            self.user_input = self.ui.load_streamlit_ui()

            if not self.user_input:
                logging.warning("User input not received from UI")
                st.error("Error: Failed to load user input from the UI.")
                return

            # Step 2: Capture user message
            user_message = st.chat_input("Enter your message:")

            if not user_message:
                logging.info("No user message entered yet")
                return

            logging.info(f"User message received: {user_message}")

            # Step 3: Initialize LLM
            self._initialize_llm()

            if not self.model:
                logging.error("LLM model initialization failed")
                st.error("Error: LLM model could not be initialized")
                return

            # Step 4: Build Graph
            self._build_graph()

            if not self.graph:
                logging.error("Graph building failed")
                st.error("Error: Graph setup failed")
                return

            # Step 5: Execute and display result
            self._display_result(user_message)

        except Exception as e:
            logging.error(f"Error in application run: {str(e)}")
            raise CustomException(e, sys)