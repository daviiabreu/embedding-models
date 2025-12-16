"""
Response cleaner for voice-friendly output.

This module provides utilities to clean LLM responses for text-to-speech,
removing formatting that sounds bad when spoken aloud:
- URLs (sound terrible: "h t t p colon slash slash...")
- Markdown formatting (**, __, *, _, ##)
- Code blocks and inline code
- Email addresses (awkward to pronounce)
- Special characters meant for visual formatting
"""

import re
from typing import NamedTuple

from .logging_config import get_logger

logger = get_logger(__name__)


class CleaningResult(NamedTuple):
    """Result of cleaning a response."""

    text: str
    changes_made: list[str]
    original_length: int
    cleaned_length: int


def clean_for_voice(text: str, aggressive: bool = False) -> CleaningResult:
    """
    Clean text for text-to-speech output.

    Removes or replaces elements that sound bad when spoken:
    - URLs are removed or replaced with natural descriptions
    - Markdown formatting is stripped
    - Code blocks are removed
    - Special characters are cleaned

    Args:
        text: The text to clean
        aggressive: If True, removes more content (like email addresses)

    Returns:
        CleaningResult with cleaned text and list of changes made

    Examples:
        >>> result = clean_for_voice("Visite https://inteli.edu.br para mais info")
        >>> result.text
        'Visite o site do Inteli para mais info'

        >>> result = clean_for_voice("Isso e **muito** importante!")
        >>> result.text
        'Isso e muito importante!'
    """
    if not text:
        return CleaningResult(
            text="",
            changes_made=[],
            original_length=0,
            cleaned_length=0,
        )

    original = text
    changes = []

    # 1. Remove URLs and replace with natural language
    url_pattern = r"https?://(?:www\.)?([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)(?:/[^\s]*)?"
    urls_found = re.findall(url_pattern, text)
    if urls_found:
        # Replace Inteli URLs with natural description
        text = re.sub(
            r"https?://(?:www\.)?inteli\.edu\.br[^\s]*",
            "o site do Inteli",
            text,
        )
        # Replace other URLs with generic description
        text = re.sub(url_pattern, "o site", text)
        changes.append(f"Removed {len(urls_found)} URL(s)")

    # 2. Remove www. prefixed URLs without http
    www_pattern = r"www\.[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+[^\s]*"
    www_found = re.findall(www_pattern, text)
    if www_found:
        text = re.sub(www_pattern, "o site", text)
        changes.append(f"Removed {len(www_found)} www URL(s)")

    # 3. Remove markdown bold (**text** or __text__)
    bold_pattern = r"\*\*([^*]+)\*\*|__([^_]+)__"
    bold_found = re.findall(bold_pattern, text)
    if bold_found:
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        changes.append("Removed markdown bold formatting")

    # 4. Remove markdown italic (*text* or _text_)
    # Be careful not to remove asterisks used for other purposes
    italic_pattern = r"(?<!\*)\*(?!\*)([^*]+)(?<!\*)\*(?!\*)"
    if re.search(italic_pattern, text):
        text = re.sub(italic_pattern, r"\1", text)
        changes.append("Removed markdown italic formatting")

    # Single underscores for italic
    text = re.sub(r"(?<!_)_(?!_)([^_]+)(?<!_)_(?!_)", r"\1", text)

    # 5. Remove markdown headers (# ## ### etc.)
    header_pattern = r"^#{1,6}\s+"
    if re.search(header_pattern, text, re.MULTILINE):
        text = re.sub(header_pattern, "", text, flags=re.MULTILINE)
        changes.append("Removed markdown headers")

    # 6. Remove code blocks
    code_block_pattern = r"```[\s\S]*?```"
    if re.search(code_block_pattern, text):
        text = re.sub(code_block_pattern, "", text)
        changes.append("Removed code blocks")

    # 7. Remove inline code
    inline_code_pattern = r"`([^`]+)`"
    if re.search(inline_code_pattern, text):
        text = re.sub(inline_code_pattern, r"\1", text)
        changes.append("Removed inline code formatting")

    # 8. Remove markdown list markers at start of lines
    list_pattern = r"^[\s]*[-*+]\s+"
    if re.search(list_pattern, text, re.MULTILINE):
        text = re.sub(list_pattern, "", text, flags=re.MULTILINE)
        changes.append("Removed list markers")

    # 9. Remove numbered list markers
    numbered_pattern = r"^\s*\d+\.\s+"
    if re.search(numbered_pattern, text, re.MULTILINE):
        text = re.sub(numbered_pattern, "", text, flags=re.MULTILINE)
        changes.append("Removed numbered list markers")

    # 10. Clean up email addresses (aggressive mode)
    if aggressive:
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails_found = re.findall(email_pattern, text)
        if emails_found:
            text = re.sub(email_pattern, "o email de contato", text)
            changes.append(f"Replaced {len(emails_found)} email address(es)")

    # 11. Convert literal \n to actual newlines, then clean excessive whitespace
    # This handles cases where the model outputs escaped newlines as text
    text = text.replace("\\n", " ")  # Replace literal \n with space for voice
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()

    # 12. Remove leftover markdown artifacts
    text = re.sub(r"~~([^~]+)~~", r"\1", text)  # Strikethrough
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # Links [text](url)

    # Log if significant changes were made
    if changes:
        logger.info(
            "response_cleaned",
            original_length=len(original),
            cleaned_length=len(text),
            changes=changes,
        )

    return CleaningResult(
        text=text,
        changes_made=changes,
        original_length=len(original),
        cleaned_length=len(text),
    )


def ensure_voice_friendly(text: str) -> str:
    """
    Simple wrapper that returns only the cleaned text.

    Use this when you just need the cleaned text without metadata.

    Args:
        text: Text to clean

    Returns:
        Cleaned text ready for TTS
    """
    return clean_for_voice(text).text


def validate_voice_friendly(text: str) -> tuple[bool, list[str]]:
    """
    Check if text is voice-friendly without modifying it.

    Args:
        text: Text to validate

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for URLs
    if re.search(r"https?://", text) or re.search(r"www\.", text):
        issues.append("Contains URLs")

    # Check for markdown bold
    if re.search(r"\*\*[^*]+\*\*", text):
        issues.append("Contains markdown bold (**)")

    # Check for markdown headers
    if re.search(r"^#{1,6}\s+", text, re.MULTILINE):
        issues.append("Contains markdown headers (#)")

    # Check for code blocks
    if re.search(r"```", text):
        issues.append("Contains code blocks")

    # Check for inline code
    if re.search(r"`[^`]+`", text):
        issues.append("Contains inline code")

    # Check for markdown links
    if re.search(r"\[([^\]]+)\]\([^)]+\)", text):
        issues.append("Contains markdown links")

    # Check for literal \n (escaped newlines as text)
    if "\\n" in text:
        issues.append("Contains literal \\n characters")

    return len(issues) == 0, issues


__all__ = [
    "clean_for_voice",
    "ensure_voice_friendly",
    "validate_voice_friendly",
    "CleaningResult",
]
