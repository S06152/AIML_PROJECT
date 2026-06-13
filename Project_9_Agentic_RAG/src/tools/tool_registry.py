from src.tools.retriever_tool import RetrieverTool
from src.tools.arxiv_tool import ArxivTool
from src.tools.tavily_tool import TavilySearchTool
from src.tools.wiki_tool import WikiTool

class ToolRegistry:

    @staticmethod
    def get_tools():
        """
        Returns a list of all available LangChain tools for the agent.
        Each tool must be a proper LangChain Tool object with name and description.
        """
        return [
            RetrieverTool.get_tool(),   # PDF document retriever (uses session state)
            ArxivTool().get_tool(),
            TavilySearchTool().get_tool(),
            WikiTool().get_tool()           
        ]