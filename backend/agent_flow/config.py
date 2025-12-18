"""Centralized configuration management."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    """LLM model configuration."""

    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    PERSPECTIVE_API_KEY: str = os.getenv("PERSPECTIVE_API_KEY", "")

    def __post_init__(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY must be set in .env")


@dataclass(frozen=True)
class RAGConfig:
    """RAG and vector database configuration."""

    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv(
        "QDRANT_COLLECTION", "inteli-documents-embeddings"
    )

    # Embedding configuration
    EMBEDDINGS_MODEL: str = os.getenv(
        "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # RAG parameters
    TOP_K: int = int(os.getenv("RAG_TOP_K", "300"))
    ADJACENT_LIMIT: int = int(os.getenv("RAG_ADJACENT_LIMIT", "10"))
    ADJACENCY_FIELD: str = os.getenv("RAG_ADJACENCY_FIELD", "adjacent_ids")
    SCORE_THRESHOLD: float | None = (
        float(os.getenv("RAG_SCORE_THRESHOLD", "0.0")) or None
    )
    INCLUDE_EMBEDDINGS: bool = os.getenv("RAG_INCLUDE_EMBEDDINGS", "false").lower() in (
        "1",
        "true",
        "yes",
    )

    def __post_init__(self):
        if not self.QDRANT_URL:
            raise ValueError("QDRANT_URL must be set in .env")


@dataclass(frozen=True)
class SafetyConfig:
    """Safety and moderation configuration."""

    # Thresholds
    TOXICITY_THRESHOLD: float = 0.7
    SIMILARITY_THRESHOLD: float = 0.7

    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = 5
    MAX_REQUESTS_PER_HOUR: int = 100

    # Input validation
    MAX_INPUT_LENGTH: int = 10_000
    MAX_CONVERSATION_HISTORY: int = 10


@dataclass(frozen=True)
class ContextConfig:
    """Context management configuration."""

    MAX_CONTEXT_TOKENS: int = 8_000
    MAX_HISTORY_MESSAGES: int = 10
    CONTEXT_FRESHNESS_DAYS: int = 90

    # Memory strategies
    MEMORY_TYPE: str = "sliding_window"  # or "selective", "summary"


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""

    model: ModelConfig
    rag: RAGConfig
    safety: SafetyConfig
    context: ContextConfig

    # Environment
    ENV: str = os.getenv("ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from environment."""
        return cls(
            model=ModelConfig(),
            rag=RAGConfig(),
            safety=SafetyConfig(),
            context=ContextConfig(),
        )


# Singleton instance
config = AppConfig.load()
