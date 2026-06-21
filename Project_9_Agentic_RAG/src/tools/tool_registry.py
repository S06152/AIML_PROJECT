from src.tools.retriever_tool import RetrieverTool
from src.tools.arxiv_tool import ArxivTool
from src.tools.tavily_tool import TavilySearchTool
from src.tools.wiki_tool import WikiTool
from src.utils.logger import logging
import warnings
warnings.filterwarnings("ignore")

class ToolRegistry:
    """
    Central registry for all tools available to the Agentic RAG system.

    Responsibilities:
        - Create and return all LangChain tools.
        - Ensure the same tool set is provided to both:
            * LLM.bind_tools()
            * LangGraph ToolNode()
        - Maintain a stable tool inventory throughout the session.

    Available Tools:
        - vector_db_retriever : Search uploaded documents.
        - wikipedia_search    : General knowledge and encyclopedic information.
        - arxiv_search        : Academic and research literature.
        - tavily_web_search   : Current and web-based information.
    """

    @staticmethod
    def get_tools():
        """
        Return all tools available to the Agentic RAG agent.

        Notes:
            - vector_db_retriever is always included.
            - If no documents have been indexed yet, the tool itself
              handles the situation gracefully and returns an informative
              message instead of raising an error.
            - Keeping the tool list constant prevents mismatches between
              the LLM's bound tools and the LangGraph ToolNode.

        Returns:
            list: Collection of LangChain tool instances.
        """

        logging.info( "Loading Agentic RAG tool registry.")

        tools = [
            RetrieverTool.get_tool(),
            WikiTool().get_tool(),
            ArxivTool().get_tool(),
            TavilySearchTool().get_tool()
        ]

        logging.info("Successfully loaded %s tools.", len(tools))

        return tools