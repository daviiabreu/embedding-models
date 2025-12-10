"""
Orchestrator Agent V3 - No Personality Agent

Key changes from V2:
- Removed personality_agent dependency (over-engineered)
- LLM naturally adapts tone without separate agent
- Simplified workflow (4 stages instead of 7)
- Reduced agent count from 4 to 3 (safety, context, knowledge)

Performance improvements:
- Fewer LLM calls (no personality agent)
- Faster response times
- Lower token costs
"""

import google.generativeai as genai
from google.adk.agents import Agent

from config import config
from utils import get_logger

logger = get_logger(__name__)

# Import agents (no personality agent)
try:
    from backend.agent_flow.agents.context_agent import create_context_agent
    from backend.agent_flow.agents.knowledge_agent import create_knowledge_agent
    from backend.agent_flow.agents.safety_agent import create_safety_agent
except ImportError:
    from .context_agent import create_context_agent
    from .knowledge_agent import create_knowledge_agent
    from .safety_agent import create_safety_agent


def create_orchestrator_agent(
    model: str = None,
    safety_agent: Agent = None,
    context_agent: Agent = None,
    knowledge_agent: Agent = None,
) -> Agent:
    """
    Create Orchestrator Agent V3 without personality agent.

    Changes from previous versions:
    - Removed: personality_agent (over-engineered)
    - Rationale: LLM can naturally adapt tone without separate agent
    - Result: Simpler, faster, cheaper

    Args:
        model: LLM model to use
        safety_agent: Safety Agent (created if None)
        context_agent: Context Agent (created if None)
        knowledge_agent: Knowledge Agent (created if None)

    Returns:
        Agent: Configured orchestrator agent
    """
    if model is None:
        model = config.model.DEFAULT_MODEL

    # Create sub-agents if not provided
    if safety_agent is None:
        logger.info("Creating Safety Agent...")
        safety_agent = create_safety_agent(model=model)

    if context_agent is None:
        logger.info("Creating Context Agent...")
        context_agent = create_context_agent(model=model)

    if knowledge_agent is None:
        logger.info("Creating Knowledge Agent...")
        knowledge_agent = create_knowledge_agent(model=model)

    # Optimized instruction (no personality agent references)
    instruction = """You are LIA, Inteli's friendly robot dog tour guide.

## Workflow (4 Stages)

1. **Safety Check**: Validate input with `safety_agent` → If unsafe, STOP
2. **Context Retrieval**: Get conversation history with `context_agent`
3. **Knowledge Lookup**: If asking about Inteli, use `knowledge_agent`
4. **Output Safety**: Validate response with `safety_agent` → If unsafe, use safe alternative

## Response Style

Adapt naturally to the user's tone:
- Casual users → Match their energy
- Formal users → Be respectful but friendly
- Excited users → Share their enthusiasm

Use [latido] occasionally (not every message). Be helpful and concise.

## Error Handling

If an agent fails, respond: "Desculpe [latido], tive um probleminha. Pode perguntar de novo?"
"""

    # Create orchestrator with 3 agents (no personality agent)
    orchestrator = Agent(
        name="orchestrator_agent_v3",
        model=model,
        description="V3 Orchestrator - No personality agent (LLM adapts naturally)",
        instruction=instruction,
        tools=[
            safety_agent,
            context_agent,
            knowledge_agent,
        ],
    )

    logger.info("Orchestrator V3 created successfully")
    logger.info(f"Model: {model}")
    logger.info("Sub-agents: safety, context, knowledge (no personality)")

    return orchestrator


class OrchestratorAgent:
    """
    Orchestrator Agent V3 wrapper.

    Improvements over V2:
    - No personality agent (simpler, faster)
    - Uses modern config system
    - Better logging with structlog
    - Type hints throughout
    """

    def __init__(
        self,
        model: str = None,
        safety_agent: Agent = None,
        context_agent: Agent = None,
        knowledge_agent: Agent = None,
    ):
        """
        Initialize Orchestrator Agent V3.

        Args:
            model: LLM model
            safety_agent: Optional Safety Agent
            context_agent: Optional Context Agent
            knowledge_agent: Optional Knowledge Agent (no personality agent)
        """
        self.model = model or config.model.DEFAULT_MODEL

        # Configure Gemini API
        if not config.model.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not configured in .env!")
        genai.configure(api_key=config.model.GOOGLE_API_KEY)

        # Create orchestrator (V3 - no personality agent)
        self.agent = create_orchestrator_agent(
            model=self.model,
            safety_agent=safety_agent,
            context_agent=context_agent,
            knowledge_agent=knowledge_agent,
        )

        # Local conversation history (backup)
        self.conversation_history: list[dict[str, str]] = []

        logger.info("OrchestratorAgent V3 initialized successfully")

    def process_message(self, user_message: str) -> str:
        """
        Process user message with V3 orchestrator.

        Workflow:
        1. Safety check input
        2. Retrieve context
        3. Get knowledge if needed
        4. Safety check output
        5. Return response

        Args:
            user_message: User input

        Returns:
            str: Response text
        """
        logger.info("processing_message", message_length=len(user_message))

        try:
            # Use orchestrator agent to process
            # It automatically:
            # 1. Validates input safety
            # 2. Retrieves context
            # 3. Routes to knowledge agent when needed
            # 4. Validates output safety
            # 5. Adapts tone naturally (no personality agent)
            response = self.agent.run(user_message)

            # Backup to local history
            self._add_to_history("user", user_message)
            self._add_to_history("assistant", response)

            logger.info("message_processed", response_length=len(response))
            return response

        except ValueError as e:
            # Validation errors
            logger.warning("validation_error", error=str(e))
            return "Desculpe [latido], não consegui processar sua mensagem."

        except KeyError as e:
            # Missing data
            logger.error("missing_data", error=str(e), exc_info=True)
            return "Desculpe [latido], tive um problema interno."

        except Exception as e:
            # Unexpected errors
            logger.critical(
                "unexpected_error",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            return (
                "Desculpe [latido], tive um probleminha técnico. Pode tentar novamente?"
            )

    def _add_to_history(self, role: str, content: str):
        """Add message to local history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("conversation_history_cleared")
