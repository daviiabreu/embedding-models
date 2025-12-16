"""Input validation utilities."""

from backend.agent_flow.config import config


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


def validate_user_input(text: str, max_length: int | None = None) -> str:
    """Validate user input before processing.

    Args:
        text: User input to validate
        max_length: Optional custom max length (uses config default if None)

    Returns:
        str: Cleaned and validated input

    Raises:
        ValidationError: If input is invalid
    """
    if max_length is None:
        max_length = config.safety.MAX_INPUT_LENGTH

    # Check if input is empty or only whitespace
    if not text or not text.strip():
        raise ValidationError("Input cannot be empty")

    # Check input length
    if len(text) > max_length:
        raise ValidationError(f"Input too long (max {max_length} characters)")

    # Check for null bytes (security issue)
    if "\x00" in text:
        raise ValidationError("Input contains null bytes")

    # Basic sanitization - check for obviously malicious patterns
    # This is a first layer of defense; safety_agent does deeper checking
    suspicious_patterns = [
        "\r\n\r\n",  # HTTP header injection attempts
    ]

    for pattern in suspicious_patterns:
        if pattern in text:
            raise ValidationError("Input contains suspicious characters")

    # Return cleaned text (strip leading/trailing whitespace)
    return text.strip()
