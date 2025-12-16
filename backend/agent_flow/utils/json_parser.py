"""
Robust JSON parser for LLM responses.

This module provides utilities to safely parse JSON from LLM-generated text,
handling common issues like:
- Markdown code blocks (```json ... ```)
- Extra text before/after JSON
- Malformed JSON with trailing commas
- Mixed content with JSON embedded in prose
"""

import json
import re
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)


class JSONParseError(Exception):
    """Raised when JSON parsing fails after all recovery attempts."""

    def __init__(
        self, message: str, original_text: str, original_error: Exception | None = None
    ):
        super().__init__(message)
        self.original_text = original_text
        self.original_error = original_error


def parse_llm_json(
    text: str,
    default: dict[str, Any] | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """
    Parse JSON from LLM-generated text with robust error handling.

    This function handles common LLM output patterns:
    1. Clean JSON: {"key": "value"}
    2. Markdown blocks: ```json\n{"key": "value"}\n```
    3. Prose with JSON: "Here is the result:\n{"key": "value"}"
    4. JSON with trailing text: {"key": "value"}\n\nNote: this is...

    Args:
        text: The LLM response text that may contain JSON
        default: Default dict to return if parsing fails (None returns empty dict)
        raise_on_error: If True, raises JSONParseError instead of returning default

    Returns:
        Parsed JSON as a dictionary

    Raises:
        JSONParseError: If raise_on_error=True and parsing fails

    Examples:
        >>> parse_llm_json('{"is_safe": true}')
        {'is_safe': True}

        >>> parse_llm_json('```json\\n{"result": "ok"}\\n```')
        {'result': 'ok'}

        >>> parse_llm_json('Here is the analysis:\\n{"score": 0.95}')
        {'score': 0.95}
    """
    if default is None:
        default = {}

    if not text or not isinstance(text, str):
        if raise_on_error:
            raise JSONParseError("Empty or invalid input text", str(text))
        return default

    text = text.strip()

    # Strategy 1: Try direct parse first (fastest path)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    extracted = _extract_from_markdown(text)
    if extracted:
        try:
            result = json.loads(extracted)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find JSON object in text using brace matching
    extracted = _find_json_object(text)
    if extracted:
        try:
            result = json.loads(extracted)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            # Try cleaning common issues
            cleaned = _clean_json_string(extracted)
            try:
                result = json.loads(cleaned)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    # Strategy 4: Regex-based JSON extraction
    extracted = _extract_json_regex(text)
    if extracted:
        try:
            result = json.loads(extracted)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # All strategies failed
    logger.warning(
        "json_parse_failed",
        text_preview=text[:200] if len(text) > 200 else text,
        text_length=len(text),
    )

    if raise_on_error:
        raise JSONParseError(
            f"Failed to parse JSON from text (length={len(text)})",
            original_text=text,
        )

    return default


def _extract_from_markdown(text: str) -> str | None:
    """Extract JSON from markdown code blocks."""
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
        r"`([\s\S]*?)`",  # `...` (inline code, less common)
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if content.startswith("{") or content.startswith("["):
                return content

    return None


def _find_json_object(text: str) -> str | None:
    """Find a JSON object using brace matching."""
    # Find the first { character
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end = start

    for i, char in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if depth == 0:
        return text[start : end + 1]

    return None


def _extract_json_regex(text: str) -> str | None:
    """Extract JSON using regex (fallback method)."""
    # Match content between first { and last }
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return None


def _clean_json_string(json_str: str) -> str:
    """Clean common JSON formatting issues from LLM output."""
    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", json_str)

    # Fix common boolean issues (True/False -> true/false)
    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = re.sub(r"\bFalse\b", "false", cleaned)
    cleaned = re.sub(r"\bNone\b", "null", cleaned)

    # Remove comments (// style)
    cleaned = re.sub(r"//[^\n]*", "", cleaned)

    return cleaned


def safe_json_parse(
    response_text: str,
    expected_fields: list[str] | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Parse JSON with field validation and defaults.

    This is a higher-level function that parses JSON and ensures
    expected fields are present with appropriate defaults.

    Args:
        response_text: LLM response containing JSON
        expected_fields: List of field names that should be in the result
        defaults: Default values for missing fields

    Returns:
        Parsed and validated dictionary

    Examples:
        >>> safe_json_parse(
        ...     '{"is_safe": true}',
        ...     expected_fields=['is_safe', 'confidence'],
        ...     defaults={'confidence': 0.5}
        ... )
        {'is_safe': True, 'confidence': 0.5}
    """
    if defaults is None:
        defaults = {}

    result = parse_llm_json(response_text, default=defaults.copy())

    # Ensure all expected fields are present
    if expected_fields:
        for field in expected_fields:
            if field not in result and field in defaults:
                result[field] = defaults[field]

    return result


__all__ = [
    "parse_llm_json",
    "safe_json_parse",
    "JSONParseError",
]
