from src.core.llm_client import GroqLLMClient
from src.core.response_chain import ResponseChain

class ChatbotService:
    """High-level chatbot orchestration."""

    def generate_response(self, query: str, api_key: str, model: str, temperature: float, max_tokens: int,) -> str:
        llm = GroqLLMClient(api_key, model, temperature, max_tokens).create()
        chain = ResponseChain(llm).build()
        return chain.invoke({"query": query})
