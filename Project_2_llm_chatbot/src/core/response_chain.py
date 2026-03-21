from langchain_core.output_parsers import StrOutputParser
from src.core.prompt import PromptFactory

class ResponseChain:
    """Builds LangChain processing pipeline."""

    def __init__(self, llm):
        self.llm = llm

    def build(self):
        return PromptFactory.create() | self.llm | StrOutputParser()
