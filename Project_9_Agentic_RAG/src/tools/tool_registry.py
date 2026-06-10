from src.tools.retriever_tool import RetrieverTool
from src.tools.arxiv_tool import ArxivTool
from src.tools.tavily_tool import TavilySearchTool
from src.tools.wiki_tool import WikiTool

class ToolRegistry:

    @staticmethod
    def get_tools():

        return [
            RetrieverTool().get_tool(),
            ArxivTool().get_tool(),
            TavilySearchTool().get_tool(),
            WikiTool().get_tool()           
        ]