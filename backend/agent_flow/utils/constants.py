"""
Centralized constants for agent flow.

All magic numbers and thresholds should be defined here for easy tuning and maintainability.
"""

# ============================================================================
# HEURISTICS CONSTANTS
# ============================================================================

# Verbosity Detection
VERBOSITY_SHORT_THRESHOLD = 10  # words
"""Messages with ≤10 words are considered 'concise'"""

VERBOSITY_LONG_THRESHOLD = 30  # words
"""Messages with >30 words are considered 'detailed'"""

# Engagement Detection
ENGAGEMENT_BASELINE_SCORE = 5.0
"""Baseline engagement score (0-10 scale)"""

ENGAGEMENT_HIGH_THRESHOLD = 7.0
"""Threshold for 'high' engagement level"""

ENGAGEMENT_MODERATE_THRESHOLD = 4.0
"""Threshold for 'moderate' engagement level (below is 'low')"""

ENGAGEMENT_LONG_MESSAGE_WORDS = 20
"""Word count that increases engagement score (+2.0)"""

ENGAGEMENT_SHORT_MESSAGE_WORDS = 5
"""Word count that decreases engagement score (-2.0)"""

ENGAGEMENT_QUALITY_MESSAGE_WORDS = 15
"""Word count threshold for 'high quality' message indicator"""

ENGAGEMENT_QUESTION_BOOST = 1.5
"""Score boost for messages with questions"""

ENGAGEMENT_ENTHUSIASM_BOOST = 1.0
"""Score boost for messages with enthusiasm (!)"""

ENGAGEMENT_CONVERSATION_BOOST = 1.0
"""Score boost for engaged conversation history (>2 messages)"""

ENGAGEMENT_CONVERSATION_HISTORY_MIN = 2
"""Minimum conversation history length to apply boost"""

# ============================================================================
# PII DETECTION CONSTANTS
# ============================================================================

# Brazilian Document Lengths
CPF_LENGTH = 11  # digits
"""Brazilian CPF (individual tax ID) length in digits"""

CNPJ_LENGTH = 14  # digits
"""Brazilian CNPJ (company tax ID) length in digits"""

# Validation Algorithm Constants
CPF_CNPJ_MODULO = 11
"""Modulo used in CPF/CNPJ check digit validation"""

CPF_CNPJ_REMAINDER_THRESHOLD = 2
"""Remainder threshold for CPF/CNPJ validation (< 2 → digit = 0)"""

# ============================================================================
# RATE LIMITING CONSTANTS
# ============================================================================

# Default rate limits (can be overridden by config)
DEFAULT_REQUESTS_PER_MINUTE = 60
"""Default rate limit: requests per minute per user"""

DEFAULT_REQUESTS_PER_HOUR = 500
"""Default rate limit: requests per hour per user"""

RATE_LIMIT_CLEANUP_WINDOW_SECONDS = 60
"""Time window for cleaning up old rate limit entries (seconds)"""

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# Input Validation
DEFAULT_MAX_INPUT_LENGTH = 10_000  # characters
"""Default maximum input length (can be overridden by config)"""

MIN_INPUT_LENGTH = 1  # characters
"""Minimum input length"""

# ============================================================================
# AGENT CONFIGURATION CONSTANTS
# ============================================================================

# Retry and Timeout
MAX_RETRY_ATTEMPTS = 3
"""Maximum number of retry attempts for agent operations"""

AGENT_TIMEOUT_SECONDS = 30
"""Default timeout for agent operations (seconds)"""

LLM_CALL_TIMEOUT_SECONDS = 60
"""Timeout for LLM API calls (seconds)"""

# ============================================================================
# LOGGING CONSTANTS
# ============================================================================

LOG_MAX_MESSAGE_LENGTH = 500
"""Maximum message length in logs (truncate after this)"""

LOG_PREVIEW_LENGTH = 50
"""Length of text preview in logs"""

# ============================================================================
# PERFORMANCE CONSTANTS
# ============================================================================

# Heuristics Performance Targets
HEURISTICS_MAX_EXECUTION_TIME_MS = 1.0
"""Target: heuristics should execute in <1ms"""

LLM_TYPICAL_LATENCY_MS = 300.0
"""Typical LLM call latency for comparison (ms)"""

HEURISTICS_SPEEDUP_FACTOR = 300
"""Expected speedup: heuristics vs LLM calls (300-400x)"""

# ============================================================================
# CACHING CONSTANTS (for future use)
# ============================================================================

DEFAULT_CACHE_TTL_SECONDS = 3600  # 1 hour
"""Default TTL for cached responses"""

SEMANTIC_SIMILARITY_THRESHOLD = 0.85
"""Similarity threshold for semantic cache matching (0-1)"""

EXACT_CACHE_SIZE = 1000
"""Maximum size of exact match cache"""

SEMANTIC_CACHE_SIZE = 500
"""Maximum size of semantic similarity cache"""
