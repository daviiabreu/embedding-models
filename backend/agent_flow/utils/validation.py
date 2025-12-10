"""Input validation utilities."""

from config import config


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


def validate_user_input(text: str) -> None:
    """Validate user input before processing.

    Raises:
        ValidationError: If input is invalid
    """
    # Check if input is empty or only whitespace
    if not text or not text.strip():
        raise ValidationError("Input cannot be empty")

    # Check input length
    if len(text) > config.safety.MAX_INPUT_LENGTH:
        raise ValidationError(
            f"Input too long (max {config.safety.MAX_INPUT_LENGTH} characters)"
        )

    # Check for null bytes (security issue)
    if "\x00" in text:
        raise ValidationError("Input contains null bytes")

    # Basic sanitization - check for obviously malicious patterns
    # This is a first layer of defense; safety_agent does deeper checking
    suspicious_patterns = [
        "\x00",  # Null bytes
        "\r\n\r\n",  # HTTP header injection attempts
    ]

    for pattern in suspicious_patterns:
        if pattern in text:
            raise ValidationError("Input contains suspicious characters")
