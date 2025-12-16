"""Custom exceptions for better error handling."""


class AgentFlowError(Exception):
    """Base exception for agent flow errors."""

    def __init__(self, message: str, user_message: str = None):
        self.message = message
        self.user_message = (
            user_message or "Desculpe, tive um probleminha. Pode tentar de novo?"
        )
        super().__init__(self.message)


class SafetyCheckError(AgentFlowError):
    """Safety check failed."""

    def __init__(self, message: str, reason: str = None):
        user_msg = "Desculpe, não posso ajudar com isso"
        self.reason = reason
        super().__init__(message, user_msg)


class RAGRetrievalError(AgentFlowError):
    """RAG retrieval failed."""

    def __init__(self, message: str):
        user_msg = "Estou com dificuldade para acessar informações agora. Pode perguntar de novo?"
        super().__init__(message, user_msg)


class RateLimitError(AgentFlowError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        """
        Initialize rate limit error.

        Args:
            message: Internal error message
            retry_after: Seconds until user can retry
        """
        user_msg = f"Você está enviando mensagens muito rápido. Aguarde {retry_after} segundos e tente novamente."
        self.retry_after = retry_after
        super().__init__(message, user_msg)


class InputValidationError(AgentFlowError):
    """Input validation failed."""

    def __init__(self, message: str):
        user_msg = f"Sua mensagem não pôde ser processada: {message}"
        super().__init__(message, user_msg)
