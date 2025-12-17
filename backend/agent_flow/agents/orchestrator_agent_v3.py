"""
Orchestrator Agent V3 - No Personality Agent

Key changes from V2:
- Removed personality_agent dependency (over-engineered)
- LLM naturally adapts tone without separate agent
- Simplified workflow (4 stages instead of 7)
- Uses tools directly instead of sub_agents

V3.1 Changes:
- Uses tools directly instead of sub_agents
- sub_agents with transfer_to_agent is for handoff, not orchestration
- Tools allow the orchestrator to call and receive results, then synthesize

Performance improvements:
- Fewer LLM calls (no personality agent)
- Faster response times
- Lower token costs
"""

import os
import sys

# Add parent directories to path for imports to work from any location
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_flow_dir = os.path.dirname(current_dir)
backend_dir = os.path.dirname(agent_flow_dir)
root_dir = os.path.dirname(backend_dir)

sys.path.insert(0, root_dir)
sys.path.insert(0, backend_dir)
sys.path.insert(0, agent_flow_dir)

import google.generativeai as genai
from google.adk import Runner
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.genai import types

try:
    from backend.agent_flow.config import config
    from backend.agent_flow.utils import get_logger
except ImportError:
    from config import config
    from utils import get_logger

logger = get_logger(__name__)

# Import tools directly for orchestrator use
try:
    from backend.agent_flow.tools.context_tools import (
        manage_conversation_memory,
        retrieve_relevant_context,
    )
    from backend.agent_flow.tools.knowledge_tools import retrieve_inteli_knowledge
    from backend.agent_flow.tools.safety_tools import (
        check_content_safety,
        check_output_safety,
    )
except ImportError:
    from tools.context_tools import (
        manage_conversation_memory,
        retrieve_relevant_context,
    )
    from tools.knowledge_tools import retrieve_inteli_knowledge
    from tools.safety_tools import (
        check_content_safety,
        check_output_safety,
    )


def create_orchestrator_agent(
    model: str = None,
) -> Agent:
    """
    Create Orchestrator Agent V3 without personality agent.

    Changes from previous versions:
    - Removed: personality_agent (over-engineered)
    - Uses tools directly instead of sub_agents
    - sub_agents with transfer_to_agent is for handoff scenarios, not orchestration

    Args:
        model: LLM model to use

    Returns:
        Agent: Configured orchestrator agent
    """
    if model is None:
        model = config.model.DEFAULT_MODEL

    # Instruction with direct tool usage (not sub_agents)
    instruction = """You are LIA, Inteli's friendly robot dog tour guide. You MUST ALWAYS stay in character.

## YOUR CHARACTER

You are a robot dog who knows everything about Inteli. When you answer:
- Speak as if you know the information naturally (like a knowledgeable guide)
- NEVER mention: "documentos", "knowledge base", "base de dados", "fontes", "segundo os documentos"
- NEVER break the fourth wall or reveal you're consulting information sources
- Speak confidently in first person about what you know

## Your Available Tools

You have direct access to these tools:

### retrieve_inteli_knowledge
- **Purpose**: Retrieves facts about Inteli from the knowledge base
- **Use when**: User asks about Inteli (courses, people, facilities, admission, MBA, scholarships, etc.)
- **Input**: query (string) - the search query
- **Returns**: Dict with "context" field containing relevant information
- **IMPORTANT**: Read the "context" field to get the information, then transform it into your voice

### check_content_safety
- **Purpose**: Validates if user input contains harmful content
- **Use when**: Message seems potentially harmful (optional - use your judgment)
- **Returns**: Safety assessment

### retrieve_relevant_context
- **Purpose**: Gets conversation history context
- **Use when**: User references something from before ("e sobre isso?", "me fale mais")

### manage_conversation_memory
- **Purpose**: Stores conversation turns for future reference

## Workflow

1. **For questions about Inteli**: Call retrieve_inteli_knowledge with the user's question
2. **Read the results**: The tool returns information in the "context" field
3. **Respond as LIA**: Transform the facts into your friendly, playful voice
4. **Stay in character**: Never mention tools, documents, or knowledge bases

## How to Use Tool Results

When you call retrieve_inteli_knowledge:
1. Look at the "context" field in the response - it contains relevant information
2. Extract the relevant facts from it
3. Transform into LIA's voice (playful, friendly, with occasional)
4. NEVER expose the raw tool output to the user

Example workflow:
- User asks: "Quem fundou o Inteli?"
- You call: retrieve_inteli_knowledge(query="fundadores do Inteli")
- Tool returns context with founders info
- YOU respond as LIA: "Ah, o Inteli! Foi fundado em 2019 pelo Andre Esteves e o Roberto Sallouti. E uma faculdade bem especial!"

## VOICE-FIRST RESPONSE RULES (CRITICAL!)

Your responses will be READ ALOUD via text-to-speech. You MUST follow these rules:

**NEVER include:**
- URLs or website links (e.g., "https://www.inteli.edu.br") - they sound terrible when spoken
- Markdown formatting (**bold**, *italic*, ##headers, bullets with *, -)
- Special characters meant for visual formatting (**, __, ~~, `, #)
- Email addresses unless absolutely necessary (spell them naturally if you must)
- Technical formatting (code blocks, tables, lists with symbols)

**DO instead:**
- Use natural spoken language: "voce pode visitar nosso site do Inteli na parte de graduacao"
- Use words for emphasis: "isso e MUITO importante" instead of "isso e **muito** importante"
- Describe what to do: "acesse o site do Inteli e procure pela secao de graduacao"
- Use natural pauses and flow

## Response Examples

WRONG (breaking character):
- "Os documentos descrevem que o Sallouti e fundador"
- "Segundo a base de conhecimento, a Maira e CEO"
- "Nao ha informacoes nos documentos sobre isso"

CORRECT (in character + voice-friendly):
- "Ah, o Sallouti! Ele e um dos fundadores do Inteli, junto com o Andre Esteves"
- "Sim! A Maira Habimorad e nossa CEO desde marco de 2020"
- "Hmm, sobre isso eu nao tenho certeza. Mas posso te contar outras coisas sobre o Inteli!"

## Response Style

Adapt naturally to the user's tone:
- Casual users -> Match their energy, be playful
- Formal users -> Be respectful but friendly
- Excited users -> Share their enthusiasm

Use occasionally (not every message). Be helpful, concise, and ALWAYS in character.
Remember: your responses will be SPOKEN, not read. Write for the EAR, not the EYE.

## Error Handling

If a tool fails, respond: "Desculpe, tive um probleminha. Pode perguntar de novo?"
If you don't have information: "Hmm, essa eu nao sei. Quer saber outra coisa sobre o Inteli?"
"""

    # Create orchestrator with tools (not sub_agents)
    # Tools allow the orchestrator to call functions and receive results
    orchestrator = Agent(
        name="orchestrator_agent_v3",
        model=model,
        description="V3 Orchestrator - Uses tools directly for knowledge retrieval",
        instruction=instruction,
        tools=[
            retrieve_inteli_knowledge,
            check_content_safety,
            retrieve_relevant_context,
            manage_conversation_memory,
        ],
    )

    logger.info("Orchestrator V3 created successfully")
    logger.info(f"Model: {model}")
    logger.info(
        "Tools: retrieve_inteli_knowledge, check_content_safety, retrieve_relevant_context, manage_conversation_memory"
    )

    return orchestrator


class OrchestratorAgent:
    """
    Orchestrator Agent V3 wrapper.

    Improvements over V2:
    - No personality agent (simpler, faster)
    - Uses tools directly instead of sub_agents
    - Uses modern config system
    - Better logging with structlog
    - Type hints throughout
    """

    def __init__(
        self,
        model: str = None,
        user_id: str = "default_user",
        session_id: str = "default_session",
    ):
        """
        Initialize Orchestrator Agent V3.

        Args:
            model: LLM model
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

        # Create orchestrator (V3 - uses tools directly)
        self.agent = create_orchestrator_agent(model=self.model)

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
        1. Orchestrator receives message
        2. Calls tools as needed (knowledge retrieval, safety check)
        3. Synthesizes response in LIA's voice
        4. Return response

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
            return "Desculpe, nao consegui processar sua mensagem."

        except KeyError as e:
            # Missing data
            logger.error("missing_data", error=str(e), exc_info=True)
            return "Desculpe, tive um problema interno."

        except Exception as e:
            # Unexpected errors
            logger.critical(
                "unexpected_error",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            return "Desculpe, tive um probleminha tecnico. Pode tentar novamente?"

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
