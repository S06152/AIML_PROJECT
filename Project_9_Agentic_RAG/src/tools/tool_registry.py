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

        vector_db_retriever is ALWAYS included (regardless of whether a PDF has
        been indexed yet) so that:
          - The tool list bound to the LLM and the ToolNode's tool list are
            always identical and stable, with no dependency on *when* during
            a Streamlit rerun this is called relative to PDF ingestion.
          - The graph/LLM never need to be rebuilt just because a document
            was indexed mid-session.

        If no document has been indexed yet, vector_db_retriever itself
        returns a clear "not available yet" message (see retriever_tool.py)
        instead of erroring — and the dynamic system prompt in agents.py
        tells the LLM not to rely on it in that case.
        """
        tools = [
            ArxivTool().get_tool(),
            TavilySearchTool().get_tool(),
            WikiTool().get_tool(),
            RetrieverTool.get_tool(),
        ]

        return tools