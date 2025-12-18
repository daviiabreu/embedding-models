"""Tests for input validation."""

import pytest

from backend.agent_flow.utils.validation import ValidationError, validate_user_input


class TestValidateUserInput:
    """Test input validation."""

    def test_valid_input(self):
        """Valid input should be cleaned and returned."""
        result = validate_user_input("  Oi, tudo bem?  ")
        assert result == "Oi, tudo bem?"

    def test_empty_input(self):
        """Empty input should raise ValidationError."""
        with pytest.raises(ValidationError, match="Input cannot be empty"):
            validate_user_input("")

    def test_whitespace_only(self):
        """Whitespace-only input should raise ValidationError."""
        with pytest.raises(ValidationError, match="Input cannot be empty"):
            validate_user_input("   \n\t  ")

    def test_input_too_long(self):
        """Input exceeding max length should raise ValidationError."""
        long_input = "x" * 10001
        with pytest.raises(
            ValidationError, match="Input too long.*max 10000 characters"
        ):
            validate_user_input(long_input, max_length=10000)

    def test_null_bytes(self):
        """Input with null bytes should raise ValidationError."""
        with pytest.raises(ValidationError, match="Input contains null bytes"):
            validate_user_input("test\x00null")

    def test_http_injection(self):
        """Input with HTTP header injection should raise ValidationError."""
        with pytest.raises(ValidationError, match="suspicious characters"):
            validate_user_input("test\r\n\r\nmalicious")

    def test_custom_max_length(self):
        """Should respect custom max length."""
        with pytest.raises(ValidationError):
            validate_user_input("x" * 101, max_length=100)

    def test_normal_multiline(self):
        """Normal multiline text should be accepted."""
        result = validate_user_input("Line 1\nLine 2\nLine 3")
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_unicode_characters(self):
        """Unicode characters should be accepted."""
        text = "Olá! Como está? 你好 🤖"
        result = validate_user_input(text)
        assert result == text

    def test_question_with_punctuation(self):
        """Questions with various punctuation should be accepted."""
        text = "Quais são os cursos? Como funciona a admissão?"
        result = validate_user_input(text)
        assert result == text

    def test_numbers_and_symbols(self):
        """Text with numbers and symbols should be accepted."""
        text = "O curso custa R$ 1.500,00 por mês (aproximadamente)."
        result = validate_user_input(text)
        assert result == text

    def test_only_spaces_between_words(self):
        """Text with only spaces between words should be cleaned."""
        text = "  Olá    mundo    Python  "
        result = validate_user_input(text)
        # Should remove leading/trailing spaces but preserve internal structure
        assert result.startswith("Olá")
        assert result.endswith("Python")


class TestValidationErrorMessages:
    """Test that validation error messages are user-friendly."""

    def test_empty_error_message(self):
        """Error message for empty input should be clear."""
        try:
            validate_user_input("")
        except ValidationError as e:
            assert "empty" in str(e).lower()

    def test_too_long_error_message(self):
        """Error message for too long input should include length info."""
        try:
            validate_user_input("x" * 10001, max_length=10000)
        except ValidationError as e:
            error_msg = str(e)
            assert "long" in error_msg.lower()
            assert "10000" in error_msg

    def test_null_byte_error_message(self):
        """Error message for null bytes should be clear."""
        try:
            validate_user_input("test\x00")
        except ValidationError as e:
            assert "null" in str(e).lower()
