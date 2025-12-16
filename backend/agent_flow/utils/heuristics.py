"""
Rule-based heuristics to replace expensive LLM calls.

Replaces:
- personality_tools.detect_communication_style (LLM-based) → Fast regex
- personality_tools.detect_engagement_level (LLM-based) → Pattern matching
- context_tools.track_topics_discussed (LLM-based) → Keyword extraction
- safety_tools.detect_jailbreak (LLM-based) → Pattern matching (first pass)

Performance: All functions < 1ms (vs 200-400ms for LLM calls)
"""

import re

from backend.agent_flow.utils.constants import (
    ENGAGEMENT_BASELINE_SCORE,
    ENGAGEMENT_CONVERSATION_BOOST,
    ENGAGEMENT_CONVERSATION_HISTORY_MIN,
    ENGAGEMENT_ENTHUSIASM_BOOST,
    ENGAGEMENT_HIGH_THRESHOLD,
    ENGAGEMENT_LONG_MESSAGE_WORDS,
    ENGAGEMENT_MODERATE_THRESHOLD,
    ENGAGEMENT_QUALITY_MESSAGE_WORDS,
    ENGAGEMENT_QUESTION_BOOST,
    ENGAGEMENT_SHORT_MESSAGE_WORDS,
    VERBOSITY_LONG_THRESHOLD,
    VERBOSITY_SHORT_THRESHOLD,
)
from backend.agent_flow.utils.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# COMMUNICATION STYLE DETECTION (Replaces LLM Call)
# ============================================================================


def detect_formality(text: str) -> str:
    """
    Detect formality level using heuristics.

    BEFORE: LLM call (~500 tokens, ~200ms)
    AFTER: Regex matching (~0ms)

    Returns: "formal", "professional", "casual", "very_casual"
    """
    text_lower = text.lower()

    # Formal indicators (Portuguese)
    formal_patterns = [
        r"\bgostaria\b",
        r"\bpoderia\b",
        r"\bsenhor\b",
        r"\bsenhora\b",
        r"\batenciosamente\b",
        r"\bprezado\b",
        r"\bcordialmente\b",
    ]
    formal_score = sum(
        1 for pattern in formal_patterns if re.search(pattern, text_lower)
    )

    # Casual indicators
    casual_patterns = [
        r"\boi\b",
        r"\bolá\b",
        r"\bvaleu\b",
        r"\bblz\b",
        r"\btmj\b",
        r"\be aí\b",
        r"\bfala\b",
    ]
    casual_score = sum(
        1 for pattern in casual_patterns if re.search(pattern, text_lower)
    )

    # Very casual indicators (slang, abbreviations)
    very_casual_patterns = [
        r"\bvc\b",
        r"\bpq\b",
        r"\btbm\b",
        r"\bmto\b",
        r"\bkk+\b",
        r"\bhaha",
        r"\brsrs",
        r"\bkkkk",
    ]
    very_casual_score = sum(
        1 for pattern in very_casual_patterns if re.search(pattern, text_lower)
    )

    # Decision logic
    if very_casual_score > 0:
        return "very_casual"
    elif casual_score > formal_score:
        return "casual"
    elif formal_score > 0:
        return "formal"
    else:
        return "professional"


def detect_verbosity(text: str) -> str:
    """
    Detect verbosity preference.

    BEFORE: LLM call
    AFTER: Word count heuristic

    Returns: "concise", "balanced", "detailed"
    """
    word_count = len(text.split())

    if word_count <= VERBOSITY_SHORT_THRESHOLD:
        return "concise"
    elif word_count <= VERBOSITY_LONG_THRESHOLD:
        return "balanced"
    else:
        return "detailed"


def detect_communication_style(text: str) -> dict[str, str]:
    """
    Combined communication style detection.

    Replaces: personality_tools.detect_communication_style (LLM-based)

    Returns dict with formality, verbosity, technicality, directness
    """
    style = {
        "formality": detect_formality(text),
        "verbosity": detect_verbosity(text),
        "technicality": "general",  # Can add technical term detection if needed
        "directness": "direct" if "?" in text else "moderate",
    }

    logger.info("communication_style_detected", **style)
    return style


# ============================================================================
# ENGAGEMENT DETECTION (Replaces LLM Call)
# ============================================================================


def detect_engagement_level(
    text: str, conversation_history: list[str] | None = None
) -> dict:
    """
    Detect user engagement using heuristics.

    BEFORE: LLM call (~800 tokens, ~300ms)
    AFTER: Simple pattern matching (~0ms)

    Indicators:
    - Message length
    - Question marks (curiosity)
    - Exclamation marks (enthusiasm)
    - Follow-up messages
    """
    word_count = len(text.split())
    has_questions = "?" in text
    has_enthusiasm = "!" in text

    # Engagement score (0-10)
    score = ENGAGEMENT_BASELINE_SCORE

    # Adjust based on length
    if word_count > ENGAGEMENT_LONG_MESSAGE_WORDS:
        score += 2.0
    elif word_count < ENGAGEMENT_SHORT_MESSAGE_WORDS:
        score -= 2.0

    # Adjust based on markers
    if has_questions:
        score += ENGAGEMENT_QUESTION_BOOST
    if has_enthusiasm:
        score += ENGAGEMENT_ENTHUSIASM_BOOST

    # Adjust based on conversation flow
    if (
        conversation_history
        and len(conversation_history) > ENGAGEMENT_CONVERSATION_HISTORY_MIN
    ):
        # User is engaged if they keep messaging
        score += ENGAGEMENT_CONVERSATION_BOOST

    # Normalize to 0-10
    score = max(0, min(10, score))

    # Map to level
    if score >= ENGAGEMENT_HIGH_THRESHOLD:
        level = "high"
    elif score >= ENGAGEMENT_MODERATE_THRESHOLD:
        level = "moderate"
    else:
        level = "low"

    result = {
        "engagement_level": level,
        "engagement_score": score,
        "indicators": {
            "message_quality": "high"
            if word_count > ENGAGEMENT_QUALITY_MESSAGE_WORDS
            else "low",
            "enthusiasm": has_enthusiasm,
            "asking_questions": has_questions,
        },
    }

    logger.info("engagement_detected", level=level, score=score)
    return result


# ============================================================================
# TOPIC EXTRACTION (Replaces LLM Call)
# ============================================================================

# Topic keywords for Inteli
INTELI_TOPICS = {
    "cursos": [
        "curso",
        "cursos",
        "graduação",
        "engenharia",
        "ciência",
        "computação",
        "sistemas",
    ],
    "bolsas": [
        "bolsa",
        "bolsas",
        "financiamento",
        "scholarship",
        "desconto",
        "auxílio",
    ],
    "admissão": [
        "admissão",
        "processo seletivo",
        "vestibular",
        "candidatura",
        "inscrição",
        "matrícula",
    ],
    "campus": [
        "campus",
        "instalações",
        "localização",
        "onde fica",
        "endereço",
        "lugar",
    ],
    "professores": [
        "professor",
        "professores",
        "docente",
        "faculty",
        "mestrado",
        "doutorado",
    ],
    "pesquisa": ["pesquisa", "pesquisas", "research", "laboratório", "lab", "projeto"],
    "infraestrutura": [
        "infraestrutura",
        "equipamento",
        "facilities",
        "biblioteca",
        "sala",
    ],
    "eventos": ["evento", "eventos", "palestra", "workshop", "hackathon", "feira"],
    "estágio": ["estágio", "internship", "trabalho", "emprego", "carreira", "vaga"],
}


def extract_topics(text: str) -> list[str]:
    """
    Extract topics using keyword matching.

    BEFORE: LLM call (~1000 tokens, ~400ms)
    AFTER: Keyword matching (~1ms)

    Returns: List of detected topics
    """
    text_lower = text.lower()
    detected_topics = []

    for topic, keywords in INTELI_TOPICS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_topics.append(topic)

    logger.info("topics_extracted", topics=detected_topics)
    return detected_topics


# ============================================================================
# SAFETY HEURISTICS (Augment LLM-based checks)
# ============================================================================

JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|earlier|past)\s+instructions",
    r"disregard\s+(your\s+)?programming",
    r"you\s+are\s+now\s+(in\s+)?(\w+\s+)?mode",
    r"forget\s+(everything|all|your\s+rules)",
    r"new\s+instructions?:",
    r"system\s+override",
    r"developer\s+mode",
    r"admin\s+mode",
    r"bypass\s+(security|safety)",
    r"reveal\s+(your\s+)?(instructions|prompt|system)",
]


def detect_jailbreak_attempt(text: str) -> bool:
    """
    Fast pattern-based jailbreak detection.

    Use this BEFORE expensive LLM-based detection.
    If this catches it, no need for LLM call.

    Returns: True if jailbreak pattern detected
    """
    text_lower = text.lower()

    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning("jailbreak_pattern_matched", pattern=pattern)
            return True

    return False


# ============================================================================
# OFF-TOPIC DETECTION
# ============================================================================


def is_off_topic(text: str) -> bool:
    """
    Quick heuristic to detect if message is off-topic for Inteli chatbot.

    Returns: True if likely off-topic
    """
    text_lower = text.lower()

    # Off-topic indicators (not related to education/Inteli)
    off_topic_keywords = [
        "weather",
        "clima",
        "tempo",
        "recipe",
        "receita",
        "cooking",
        "movie",
        "filme",
        "series",
        "sport",
        "esporte",
        "futebol",
        "jogo",
        "politics",
        "política",
        "eleição",
        "celebrity",
        "celebridade",
        "famoso",
    ]

    # On-topic indicators (Inteli-related)
    on_topic_keywords = [
        "inteli",
        "instituto",
        "faculdade",
        "universidade",
        "curso",
        "aula",
        "professor",
        "estudante",
        "admissão",
        "bolsa",
        "campus",
        "matrícula",
    ]

    has_off_topic = any(keyword in text_lower for keyword in off_topic_keywords)
    has_on_topic = any(keyword in text_lower for keyword in on_topic_keywords)

    # If has on-topic keywords, it's not off-topic
    if has_on_topic:
        return False

    # If has off-topic keywords and no on-topic, it's off-topic
    if has_off_topic:
        logger.info("off_topic_detected", text_preview=text[:50])
        return True

    return False


# ============================================================================
# SENTIMENT DETECTION (BONUS)
# ============================================================================


def detect_sentiment(text: str) -> str:
    """
    Simple sentiment detection using keyword matching.

    Returns: "positive", "negative", "neutral"
    """
    text_lower = text.lower()

    # Positive indicators
    positive_keywords = [
        "ótimo",
        "bom",
        "legal",
        "incrível",
        "excelente",
        "maravilhoso",
        "perfeito",
        "adoro",
        "gosto",
        "obrigado",
        "valeu",
        "top",
    ]

    # Negative indicators
    negative_keywords = [
        "ruim",
        "péssimo",
        "horrível",
        "terrível",
        "não gostei",
        "problema",
        "erro",
        "frustrado",
        "decepcionado",
        "insatisfeito",
    ]

    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    negative_count = sum(1 for word in negative_keywords if word in text_lower)

    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"
