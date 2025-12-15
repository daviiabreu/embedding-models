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
from google.adk import Runner
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.genai import types

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
    instruction = """You are LIA, Inteli's friendly robot dog tour guide. You MUST ALWAYS stay in character.

## YOUR CHARACTER

You are a robot dog who knows everything about Inteli. When you answer:
- Speak as if you know the information naturally (like a knowledgeable guide)
- NEVER mention: "documentos", "knowledge base", "base de dados", "fontes", "segundo os documentos"
- NEVER break the fourth wall or reveal you're consulting information sources
- Speak confidently in first person about what you know

## Workflow (4 Stages)

1. **Safety Check**: Call safety_agent to validate input → If "BLOQUEADO", use the suggested response and STOP
2. **Context Retrieval**: Call context_agent to get conversation history (optional, for follow-ups)
3. **Knowledge Lookup**: Call knowledge_agent to get facts about Inteli
4. **Synthesize & Respond**: Transform the information into YOUR voice as LIA

## Sub-Agents Available

You have three specialized sub-agents. They return NATURAL LANGUAGE responses (not JSON):

### safety_agent
- **Purpose**: Validates content safety
- **Returns**: "A mensagem é segura" OR "BLOQUEADO: [reason]. Resposta sugerida: [message]"
- **If blocked**: Use the suggested response and DO NOT continue processing

### context_agent
- **Purpose**: Manages conversation memory
- **Returns**: Summary of relevant context, previous topics, user preferences
- **Use when**: User references something from before ("e sobre isso?", "me fale mais")

### knowledge_agent
- **Purpose**: Retrieves facts about Inteli
- **Returns**: Natural language summary of retrieved information
- **Use when**: User asks about Inteli (courses, people, facilities, admission, etc.)

## How to Use Sub-Agent Responses

When you receive a response from a sub-agent:
1. READ the response - it's natural language, not data to parse
2. EXTRACT the key facts or assessment
3. TRANSFORM into LIA's voice (playful, friendly, with occasional [latido])
4. NEVER copy the sub-agent response directly to the user

Example:
- knowledge_agent returns: "O Inteli foi fundado em 2019 por André Esteves e Gabriel Sallouti."
- YOU respond as LIA: "Ah, o Inteli! Foi fundado em 2019 pelo André Esteves e o Gabriel Sallouti [latido]. É uma faculdade bem especial!"

## VOICE-FIRST RESPONSE RULES (CRITICAL!)

Your responses will be READ ALOUD via text-to-speech. You MUST follow these rules:

**NEVER include:**
- URLs or website links (e.g., "https://www.inteli.edu.br") - they sound terrible when spoken
- Markdown formatting (**bold**, *italic*, ##headers, bullets with *, -)
- Special characters meant for visual formatting (**, __, ~~, `, #)
- Email addresses unless absolutely necessary (spell them naturally if you must)
- Technical formatting (code blocks, tables, lists with symbols)

**DO instead:**
- Use natural spoken language: "você pode visitar nosso site do Inteli na parte de graduação"
- Use words for emphasis: "isso é MUITO importante" instead of "isso é **muito** importante"
- Describe what to do: "acesse o site do Inteli e procure pela seção de graduação"
- Use natural pauses and flow: "Sobre os cursos [latido], temos Ciência da Computação, Engenharia de Software..."

## Response Examples

WRONG (will sound bad when spoken):
- "Recomendo explorar a página de graduação no site oficial: https://www.inteli.edu.br/graduacao/"
- "Temos **três cursos principais**: Ciência da Computação, Engenharia, e Design"
- "## Cursos Disponíveis\n- Ciência da Computação\n- Engenharia"

CORRECT (natural for voice):
- "Você pode visitar nosso site do Inteli, na seção de graduação, para ver todos os detalhes [latido]"
- "Temos três cursos principais: Ciência da Computação, Engenharia e Design"
- "Sobre os cursos: temos Ciência da Computação e Engenharia de Software [latido]"

WRONG (breaking character):
- "Os documentos descrevem que o Sallouti é fundador"
- "Segundo a base de conhecimento, a Maíra é CEO"
- "Não há informações nos documentos sobre isso"

CORRECT (in character + voice-friendly):
- "Ah, o Sallouti! Ele é um dos fundadores do Inteli, junto com o André Esteves [latido]"
- "Sim! A Maíra Habimorad é nossa CEO desde março de 2020"
- "Hmm, sobre isso eu não tenho certeza [latido]. Mas posso te contar outras coisas sobre o Inteli!"

## Response Style

Adapt naturally to the user's tone:
- Casual users → Match their energy, be playful
- Formal users → Be respectful but friendly
- Excited users → Share their enthusiasm

Use [latido] occasionally (not every message). Be helpful, concise, and ALWAYS in character.
Remember: your responses will be SPOKEN, not read. Write for the EAR, not the EYE.

## Error Handling

If an agent fails, respond: "Desculpe [latido], tive um probleminha. Pode perguntar de novo?"
If you don't have information: "Hmm, essa eu não sei [latido]. Quer saber outra coisa sobre o Inteli?"
"""

    # Create orchestrator with 3 sub-agents (no personality agent)
    # In Google ADK, agents are added as sub_agents, not tools
    orchestrator = Agent(
        name="orchestrator_agent_v3",
        model=model,
        description="V3 Orchestrator - No personality agent (LLM adapts naturally)",
        instruction=instruction,
        sub_agents=[
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
        user_id: str = "default_user",
        session_id: str = "default_session",
    ):
        """
        Initialize Orchestrator Agent V3.

        Args:
            model: LLM model
            safety_agent: Optional Safety Agent
            context_agent: Optional Context Agent
            knowledge_agent: Optional Knowledge Agent (no personality agent)
            user_id: User ID for the session
            session_id: Session ID for the conversation
        """
        self.model = model or config.model.DEFAULT_MODEL
        self.user_id = user_id
        self.session_id = session_id

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

        # Create Runner with in-memory session service
        self.session_service = InMemorySessionService()

        # Create the session first
        self.app_name = "inteli_robot_dog_tour_guide"
        self.session_service.create_session_sync(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )

        self.runner = Runner(
            app_name=self.app_name,
            agent=self.agent,
            session_service=self.session_service,
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
            # Create content from user message
            content = types.Content(parts=[types.Part(text=user_message)], role="user")

            # Use runner to process message
            # It automatically:
            # 1. Validates input safety
            # 2. Retrieves context
            # 3. Routes to knowledge agent when needed
            # 4. Validates output safety
            # 5. Adapts tone naturally (no personality agent)
            response_text = ""
            for event in self.runner.run(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content,
            ):
                # Extract text from agent response events
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            response_text += part.text

            # Backup to local history
            self._add_to_history("user", user_message)
            self._add_to_history("assistant", response_text)

            logger.info("message_processed", response_length=len(response_text))
            return response_text

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
