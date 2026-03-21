class AppSettings:
    """Centralized application settings."""

    APP_TITLE = "Enterprise Q & A Chatbot with GROQ"

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 300
    MIN_TOKENS = 50
    MAX_TOKENS = 2000

    AVAILABLE_MODELS = [
        "openai/gpt-oss-safeguard-20b", 
        "qwen/qwen3-32b", 
        "llama-3.3-70b-versatile", 
        "meta-llama/llama-4-scout-17b-16e-instruct", 
        "meta-llama/llama-guard-4-12b", 
        "openai/gpt-oss-20b"
    ]
