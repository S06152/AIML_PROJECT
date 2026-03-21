class InputValidator:
    """Validates user inputs."""

    @staticmethod
    def is_valid(text: str) -> bool:
        return bool(text and text.strip())
