"""
Enhanced PII (Personally Identifiable Information) detection.

Improvements over basic regex:
- More comprehensive patterns
- Better obfuscation detection ("john at example dot com")
- Validation of detected PII (e.g., CPF algorithm)
- Both detection and masking capabilities
"""

import re

from backend.agent_flow.utils.constants import (
    CNPJ_LENGTH,
    CPF_CNPJ_MODULO,
    CPF_CNPJ_REMAINDER_THRESHOLD,
    CPF_LENGTH,
)
from backend.agent_flow.utils.logging_config import get_logger

logger = get_logger(__name__)


# Enhanced PII patterns for Brazilian context
# Pre-compiled for 10-20% performance improvement
PII_PATTERNS = {
    "email": [
        # Standard email (with unicode support for names like joão)
        re.compile(
            r"[A-Za-zÀ-ÿ0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}", re.IGNORECASE
        ),
        # Email with spaces around @ and dot
        re.compile(
            r"[A-Za-zÀ-ÿ0-9._%+-]+\s+@\s+[A-Za-z0-9.-]+\s+\.\s+[A-Z|a-z]{2,}",
            re.IGNORECASE,
        ),
        # Obfuscated (john at example dot com)
        re.compile(
            r"[A-Za-zÀ-ÿ0-9._%+-]+\s+(?:at|arroba)\s+[A-Za-z0-9.-]+\s+(?:dot|ponto)\s+[A-Z|a-z]{2,}",
            re.IGNORECASE,
        ),
    ],
    "phone_br": [
        # Brazilian phone: +55 (11) 98765-4321
        re.compile(
            r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}\b", re.IGNORECASE
        ),
        # With spaces: 11 98765 4321
        re.compile(r"\b\d{2}\s+\d{4,5}\s+\d{4}\b", re.IGNORECASE),
    ],
    "cpf": [
        # Standard CPF: 123.456.789-01
        re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", re.IGNORECASE),
        # With label
        re.compile(r"\bcpf:?\s*\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", re.IGNORECASE),
    ],
    "credit_card": [
        # 4-digit groups
        re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", re.IGNORECASE),
    ],
    "cep": [
        # Brazilian postal code: 01234-567
        re.compile(r"\b\d{5}-?\d{3}\b", re.IGNORECASE),
        # With label
        re.compile(r"\bcep:?\s*\d{5}-?\d{3}\b", re.IGNORECASE),
    ],
    "cnpj": [
        # Brazilian company ID: 12.345.678/0001-90
        re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b", re.IGNORECASE),
        # With label
        re.compile(r"\bcnpj:?\s*\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b", re.IGNORECASE),
    ],
}


def detect_pii(text: str) -> dict[str, list[tuple[str, int, int]]]:
    """
    Detect PII in text with enhanced patterns.

    Uses pre-compiled regex patterns for optimal performance (10-20% faster).

    Args:
        text: Text to scan for PII

    Returns:
        Dict mapping PII type to list of (matched_text, start, end) tuples
    """
    detected: dict[str, list[tuple[str, int, int]]] = {}

    for pii_type, patterns in PII_PATTERNS.items():
        matches = []
        seen_positions = set()  # Track positions to avoid duplicates

        for pattern in patterns:
            # Patterns are now pre-compiled, use finditer directly
            for match in pattern.finditer(text):
                matched_text = match.group()
                start, end = match.start(), match.end()

                # Additional validation for CPF
                if pii_type == "cpf" and not validate_cpf(matched_text):
                    # Skip invalid CPF (reduces false positives)
                    continue

                # Additional validation for CNPJ
                if pii_type == "cnpj" and not validate_cnpj(matched_text):
                    # Skip invalid CNPJ (reduces false positives)
                    continue

                # Check for overlapping matches (avoid duplicates)
                position_key = (start, end)
                if position_key not in seen_positions:
                    seen_positions.add(position_key)
                    matches.append((matched_text, start, end))

        if matches:
            detected[pii_type] = matches

    if detected:
        total_count = sum(len(v) for v in detected.values())
        logger.warning("pii_detected", types=list(detected.keys()), count=total_count)

    return detected


def validate_cpf(cpf: str) -> bool:
    """
    Validate Brazilian CPF number using check digit algorithm.

    This reduces false positives by checking if the CPF is mathematically valid.

    Args:
        cpf: CPF string (with or without formatting)

    Returns:
        bool: True if CPF is valid
    """
    # Remove non-digits
    cpf = re.sub(r"\D", "", cpf)

    if len(cpf) != CPF_LENGTH:
        return False

    # Check for known invalid CPFs (all same digit)
    if cpf == cpf[0] * CPF_LENGTH:
        return False

    # Validate check digits
    def calculate_digit(cpf_partial: str, weights: list[int]) -> int:
        total = sum(int(digit) * weight for digit, weight in zip(cpf_partial, weights))
        remainder = total % CPF_CNPJ_MODULO
        return (
            0
            if remainder < CPF_CNPJ_REMAINDER_THRESHOLD
            else CPF_CNPJ_MODULO - remainder
        )

    weights_first = list(range(10, 1, -1))
    weights_second = list(range(11, 1, -1))

    first_digit = calculate_digit(cpf[:9], weights_first)
    second_digit = calculate_digit(cpf[:10], weights_second)

    return cpf[-2:] == f"{first_digit}{second_digit}"


def validate_cnpj(cnpj: str) -> bool:
    """
    Validate Brazilian CNPJ number using check digit algorithm.

    This reduces false positives by checking if the CNPJ is mathematically valid.

    Args:
        cnpj: CNPJ string (with or without formatting)

    Returns:
        bool: True if CNPJ is valid
    """
    # Remove non-digits
    cnpj = re.sub(r"\D", "", cnpj)

    if len(cnpj) != CNPJ_LENGTH:
        return False

    # Check for known invalid CNPJs (all same digit)
    if cnpj == cnpj[0] * CNPJ_LENGTH:
        return False

    # Validate check digits
    def calculate_digit(cnpj_partial: str, weights: list[int]) -> int:
        total = sum(int(digit) * weight for digit, weight in zip(cnpj_partial, weights))
        remainder = total % CPF_CNPJ_MODULO
        return (
            0
            if remainder < CPF_CNPJ_REMAINDER_THRESHOLD
            else CPF_CNPJ_MODULO - remainder
        )

    # First check digit
    weights_first = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    first_digit = calculate_digit(cnpj[:12], weights_first)

    # Second check digit
    weights_second = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    second_digit = calculate_digit(cnpj[:13], weights_second)

    return cnpj[-2:] == f"{first_digit}{second_digit}"


def mask_pii(text: str, detected_pii: dict | None = None) -> str:
    """
    Mask detected PII in text.

    Args:
        text: Original text
        detected_pii: Pre-detected PII (if None, will detect)

    Returns:
        str: Text with PII masked
    """
    if detected_pii is None:
        detected_pii = detect_pii(text)

    if not detected_pii:
        return text

    masked = text
    offset = 0

    # Sort all matches by position
    all_matches: list[tuple[int, int, str, str]] = []
    for pii_type, matches in detected_pii.items():
        for match_text, start, end in matches:
            all_matches.append((start, end, match_text, pii_type))

    all_matches.sort()

    # Mask from left to right
    for start, end, match_text, pii_type in all_matches:
        # Adjust for previous replacements
        adj_start = start + offset
        adj_end = end + offset

        # Create appropriate mask
        if pii_type == "email":
            # Show first char + ***@domain
            if "@" in match_text:
                parts = match_text.split("@")
                mask = f"{parts[0][0]}***@{parts[1]}"
            else:
                mask = "***@***"
        elif pii_type in ("cpf", "cnpj", "phone_br", "credit_card"):
            # Show first 2 and last 2 digits
            if len(match_text) > 4:
                mask = match_text[:2] + "*" * (len(match_text) - 4) + match_text[-2:]
            else:
                mask = "*" * len(match_text)
        else:
            # Generic masking
            mask = "*" * len(match_text)

        # Replace in text
        masked = masked[:adj_start] + mask + masked[adj_end:]
        offset += len(mask) - len(match_text)

    return masked


def has_pii(text: str) -> bool:
    """
    Quick check if text contains PII.

    Args:
        text: Text to check

    Returns:
        bool: True if PII detected
    """
    detected = detect_pii(text)
    return len(detected) > 0


def get_pii_types(text: str) -> list[str]:
    """
    Get list of PII types detected in text.

    Args:
        text: Text to check

    Returns:
        list: PII types detected (e.g., ["email", "cpf"])
    """
    detected = detect_pii(text)
    return list(detected.keys())


def sanitize_text(text: str, mask_pii_data: bool = True) -> tuple[str, bool]:
    """
    Sanitize text by detecting and optionally masking PII.

    Args:
        text: Text to sanitize
        mask_pii_data: If True, mask PII; if False, just detect

    Returns:
        tuple: (sanitized_text, has_pii)
    """
    detected = detect_pii(text)
    has_pii_flag = len(detected) > 0

    if has_pii_flag and mask_pii_data:
        sanitized = mask_pii(text, detected)
    else:
        sanitized = text

    if has_pii_flag:
        logger.info(
            "text_sanitized",
            pii_types=list(detected.keys()),
            masked=mask_pii_data,
        )

    return sanitized, has_pii_flag
