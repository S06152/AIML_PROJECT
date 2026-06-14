from src.tools.retriever_tool import RetrieverTool
from src.tools.arxiv_tool import ArxivTool
from src.tools.tavily_tool import TavilySearchTool
from src.tools.wiki_tool import WikiTool
import streamlit as st

class ToolRegistry:

    @staticmethod
    def get_tools():
        """
        Returns a list of all available LangChain tools for the agent.
        Each tool must be a proper LangChain Tool object with name and description.
        """
        tools = [
            ArxivTool().get_tool(),
            TavilySearchTool().get_tool(),
            WikiTool().get_tool()
        ]
 
        if "vector_retriever" in st.session_state:
            tools.append(RetrieverTool.get_tool())

        return tools