import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

class WikiTool:

    def __init__(self):

        try:
            api_wrapper_wiki = WikipediaAPIWrapper(top_k_results = 2, doc_content_chars_max = 3000)
            self._tool = WikipediaQueryRun(api_wrapper = api_wrapper_wiki)
            logging.info("Wikipedia Tool initialized")

        except Exception as e:
            logging.exception("Failed to initialize Wikipedia  Tool.")
            raise CustomException(e, sys)


    def get_tool(self):
        return self._tool