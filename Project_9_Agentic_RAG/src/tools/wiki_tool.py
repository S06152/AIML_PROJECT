import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

class WikiTool:

    def __init__(self):

        try:
            self._api_wrapper = WikipediaAPIWrapper(top_k_results = 2, doc_content_chars_max = 3000)
            logging.info("Wikipedia Tool initialized")

        except Exception as e:
            logging.exception("Failed to initialize Wikipedia  Tool.")
            raise CustomException(e, sys)

    def get_tool(self):
        api_wrapper = self._api_wrapper

        @tool
        def wikipedia_search(query: str) -> str:
            """Search Wikipedia for encyclopaedic facts, definitions, historical events, 
            biographies, scientific concepts, or general knowledge. Use this tool when the 
            user asks about well-known topics, people, places, or concepts that would be 
            found in an encyclopedia."""
            try:
                logging.info(f"wikipedia_search tool invoked with query: {query}")
                result = api_wrapper.run(query)
                return result if result else "No Wikipedia results found for the given query."
            except Exception as e:
                logging.exception("Error in wikipedia_search tool.")
                return f"Error searching Wikipedia: {str(e)}"

        return wikipedia_search