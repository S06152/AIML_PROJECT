import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
import warnings
warnings.filterwarnings('ignore')

class ArxivTool:

    def __init__(self):

        try:
            api_wrapper_arxiv = ArxivAPIWrapper(top_k_results = 3, load_max_docs = 3, doc_content_chars_max = 4000)

            self._tool = ArxivQueryRun(api_wrapper = api_wrapper_arxiv)

            logging.info("Arxiv Tool initialized")

        except Exception as e:
            logging.exception("Failed to initialize Arxiv Tool.")
            raise CustomException(e, sys)

    def get_tool(self):
        return self._tool