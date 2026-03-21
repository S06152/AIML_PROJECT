import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_groq import ChatGroq
import streamlit as st

class GROQLLM:
    """
    GROQLLM is responsible for initializing and returning
    the Groq LLM model based on user inputs or environment variables.

    Attributes:
        user_contols_input (dict): Dictionary containing user inputs like
                                  API key and selected model.
    """
    def __init__(self, user_contols_input: dict):
        try:
            logging.info("Initializing GROQLLM class")
            self.user_contols_input = user_contols_input

        except Exception as e:
            logging.error("Error during GROQLLM initialization")
            raise CustomException(e, sys)
    
    def get_llm_model(self):
        """
        Initializes and returns the ChatGroq model.

        Priority:
        1. User-provided API key
        2. Environment variable

        Returns:
            llm: ChatGroq model instance
        """
        try:
            logging.info("Fetching GROQ LLM configuration")

            # Fetch API key 
            groq_api_key = self.user_contols_input["GROQ_API_KEY"]

            # Validate API key
            if not groq_api_key:
                logging.warning("GROQ API key not provided")
                st.error("Please enter the Groq API Key")
                return None

            # Fetch LLM model 
            selected_groq_model = self.user_contols_input["selected_groq_model"]

            if not selected_groq_model:
                logging.warning("No Groq model selected")
                st.error("Please select a Groq model")
                return None

            logging.info(f"Initializing ChatGroq model: {selected_groq_model}")

            # Initialize LLM
            llm = ChatGroq(api_key = groq_api_key, model = selected_groq_model)

            logging.info("ChatGroq model initialized successfully")

            return llm

        except Exception as e:
            logging.error("Error while creating GROQ LLM model")
            raise CustomException(e, sys)


