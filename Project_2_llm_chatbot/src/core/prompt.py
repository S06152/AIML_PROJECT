from langchain_core.prompts import ChatPromptTemplate

class PromptFactory:
    """Creates prompt templates."""

    @staticmethod
    def create() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Answer clearly and concisely."),
                ("user", "{query}")
            ]
        )
