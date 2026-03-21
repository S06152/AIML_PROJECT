from langchain_groq import ChatGroq

class GroqLLMClient:
    """Groq LLM client wrapper."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def create(self) -> ChatGroq:
        return ChatGroq(groq_api_key = self.api_key, model = self.model, temperature = self.temperature, max_tokens = self.max_tokens)
