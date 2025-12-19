"""
Chat Service V3 - Phase 2 Integration

Improvements over V2:
- Uses Orchestrator V3 (no personality agent)
- Rate limiting per user
- Enhanced PII detection
- Better error handling with specific exceptions
- Metrics collection
- Structured logging
- Request ID tracking for debugging
"""

import uuid

from backend.agent_flow.utils import (
    RateLimitError,
    SafetyCheckError,
    ValidationError,
    clean_for_voice,
    configure_logging,
    get_all_cache_stats,
    get_logger,
    get_rate_limiter,
    has_pii,
    metrics,
    validate_user_input,
)

# Import V3 orchestrator
try:
    from backend.agent_flow.agents.orchestrator_agent_v3 import OrchestratorAgent
except ImportError:
    from .agents.orchestrator_agent_v3 import OrchestratorAgent

# Configure logging
configure_logging()
logger = get_logger(__name__)


class ChatService:
    """
    Chat Service V3 with Phase 2 enhancements.

    Features:
    - Rate limiting (60/min, 500/hour per user)
    - Input validation (length, null bytes, etc.)
    - PII detection and warning
    - Orchestrator V3 (no personality agent)
    - Comprehensive error handling
    - Metrics collection
    """

    def __init__(self):
        """Initialize Chat Service V3."""
        logger.info("chat_service_v3_initializing")

        # Initialize orchestrator V3
        self.orchestrator = OrchestratorAgent()

        # Get rate limiter singleton
        self.rate_limiter = get_rate_limiter()

        logger.info("chat_service_v3_initialized")

    def give_response(
        self, prompt: str, user_id: str | None = None, request_id: str | None = None
    ) -> str:
        """
        Process user prompt with Phase 2 enhancements.

        Args:
            prompt: User input
            user_id: User identifier for rate limiting (optional, defaults to "anonymous")
            request_id: Optional request ID for tracking (auto-generated if None)

        Returns:
            str: Response text (always user-safe)
        """
        # Default user_id if not provided
        if user_id is None:
            user_id = "anonymous"

        # Generate request ID for tracking
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Track request start
        logger.info("request_started", request_id=request_id, user_id=user_id)

        try:
            # 1. Check rate limit FIRST
            self.rate_limiter.check_rate_limit(user_id)
            logger.debug("rate_limit_passed", user_id=user_id)

            # 2. Validate input
            validated_prompt = validate_user_input(prompt)
            logger.info(
                "request_received",
                user_id=user_id,
                prompt_length=len(validated_prompt),
            )

            # 3. Check for PII (warn but don't block - safety_agent will handle)
            if has_pii(validated_prompt):
                logger.warning(
                    "pii_detected_in_input",
                    user_id=user_id,
                )
                # Note: We don't block here - safety_agent will handle PII properly

            # 4. Process with orchestrator V3
            raw_response = self.orchestrator.process_message(validated_prompt)

            # 5. Clean response for voice output (remove URLs, markdown, etc.)
            logger.debug("raw_response_before_cleaning", raw=repr(raw_response))
            cleaning_result = clean_for_voice(raw_response)
            response = cleaning_result.text
            logger.debug("response_after_cleaning", cleaned=repr(response))

            # Log if cleaning made changes
            if cleaning_result.changes_made:
                logger.info(
                    "response_cleaned_for_voice",
                    request_id=request_id,
                    changes=cleaning_result.changes_made,
                    original_length=cleaning_result.original_length,
                    cleaned_length=cleaning_result.cleaned_length,
                )

            # 6. Log success
            logger.info(
                "request_completed",
                request_id=request_id,
                user_id=user_id,
                response_length=len(response),
            )

            return response

        except RateLimitError as e:
            # Rate limit exceeded
            logger.warning(
                "rate_limit_exceeded", request_id=request_id, user_id=user_id
            )
            return e.user_message

        except ValidationError as e:
            # Input validation failed
            logger.warning(
                "validation_failed",
                request_id=request_id,
                user_id=user_id,
                error=str(e),
            )
            return e.user_message

        except SafetyCheckError as e:
            # Safety violation
            logger.warning(
                "safety_violation",
                request_id=request_id,
                user_id=user_id,
                reason=getattr(e, "reason", "unknown"),
            )

            return e.user_message

        except Exception as e:
            # Unexpected error - log with full trace but return safe message
            logger.error(
                "unexpected_error",
                request_id=request_id,
                user_id=user_id,
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            return "Desculpe, tive um problema inesperado. Pode tentar novamente?"

    def gives_context_to_llm(self, text: str) -> str:
        text = text.replace('engcomp', 'Engenharia da Computação')
        text = text.replace('engsoft', 'Engenharia de Software')
        text = text.replace('si', 'Sistemas de Informação')
        text = text.replace('cc', 'Ciência da Computação')

        return text

    def get_usage(self, user_id: str) -> dict:
        """
        Get rate limit usage for a user.

        Args:
            user_id: User identifier

        Returns:
            dict: Usage statistics
        """
        return self.rate_limiter.get_usage(user_id)

    def reset_rate_limit(self, user_id: str):
        """
        Reset rate limit for a specific user (admin function).

        Args:
            user_id: User identifier
        """
        self.rate_limiter.reset_user(user_id)
        logger.info("rate_limit_reset_by_admin", user_id=user_id)

    def get_cache_stats(self) -> dict:
        """
        Get statistics for all caches (RAG, embedding, LLM).

        Returns:
            dict: Cache statistics including hit rates and sizes
        """
        return get_all_cache_stats()
